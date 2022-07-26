from select import select
import sys
import torch
import logging
import time
from hyperpyyaml import load_hyperpyyaml
import yaml
import sentencepiece as sp
from model import module
from model.module import linear

from utils.distributed import run_on_main, ddp_init_group
from data.dataio.dataloader import make_dataloader
from data.dataio.dataset import dataio_prepare
from utils.utils import check_gradients, update_average, create_experiment_directory, parse_arguments, init_log, parse_args
from utils.parameter_transfer import load_torch_model, load_spm
from model.module.convolution import ConvolutionFrontEnd
from model.TransformerASR import TransformerASR
from model.module.linear import Linear
from data.processing.features import InputNormalization
from utils.checkpoints import Checkpointer
from model.TransformerLM import TransformerLM
from model.decoders.seq2seq import S2STransformerBeamSearch


def train_epoch(model, optimizer, train_set, epoch, hparams):
    logger = logging.getLogger("train")
    if train_set.sampler is not None and hasattr(train_set.sampler, "set_epoch"):
        train_set.sampler.set_epoch(epoch)
    model.train()
    
    step = 0
    nonfinite_count = 0
    total_step = len(train_set)
    avg_train_loss = 0.0
    epoch_start_time = time.time()
    for batch in train_set:
        step += 1
        step_start_time = time.time()
        should_step = step % hparams["grad_accumulation_factor"] == 0
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos
        feats = hparams["compute_features"](wavs)
        feats = model["normalize"](feats, wav_lens, epoch=epoch)
        feats = hparams["augmentation"](feats)

        src = model["CNN"](feats)
        enc_out, pred = model["Transformer"](src, tokens_bos, wav_lens, pad_idx=hparams["pad_index"])

        logits = model["ctc_lin"](enc_out)
        p_ctc = hparams["log_softmax"](logits)

        pred = model["seq_lin"](pred)
        p_seq = hparams["log_softmax"](pred)

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        loss_seq = hparams["seq_cost"](p_seq, tokens_eos, length=tokens_eos_lens).sum()
        loss_ctc = hparams["ctc_cost"](p_ctc, tokens, wav_lens, tokens_lens).sum()
        loss = (hparams["ctc_weight"] * loss_ctc + (1 - hparams["ctc_weight"]) * loss_seq)
        (loss / hparams["grad_accumulation_factor"]).backward()

        if should_step:
            is_loss_finite, nonfinite_count = check_gradients(model, loss, hparams["max_grad_norm"], nonfinite_count)
            if is_loss_finite:
                optimizer.step()
            optimizer.zero_grad()
            hparams["noam_annealing"](optimizer)

        train_loss = loss.detach().cpu()
        avg_train_loss = update_average(train_loss, avg_train_loss, step)
        logger.info(f"epoch: {epoch}, step: {step}|{total_step}, time: {(time.time()-step_start_time):.2f}s, loss: {train_loss}, avg_loss: {avg_train_loss:.4f}, lr: {hparams['noam_annealing'].current_lr}")

        if step==7:
            break
    logger.info(f"epoch: {epoch}, time: {(time.time()-epoch_start_time):.2f}s, avg_loss: {avg_train_loss:.4f}")


def evaluate(model, valid_set, epoch, hparams, tokenizer):
    logger = logging.getLogger("evaluate")
    acc_metric = hparams["acc_computer"]()
    wer_metric = hparams["error_rate_computer"]()
    model.eval()
    avg_valid_loss = 0.0
    total_step = len(valid_set)
    step = 0
    eval_start_time = time.time()
    with torch.no_grad():
        for batch in valid_set:
            step += 1
            step_start_time = time.time()
            wavs, wav_lens = batch.sig
            tokens_bos, _ = batch.tokens_bos
            feats = hparams["compute_features"](wavs)
            feats = model["normalize"](feats, wav_lens, epoch=epoch)

            src = model["CNN"](feats)
            enc_out, pred = model["Transformer"](src, tokens_bos, wav_lens, pad_idx=hparams["pad_index"])

            logits = model["ctc_lin"](enc_out)
            p_ctc = hparams["log_softmax"](logits)

            pred = model["seq_lin"](pred)
            p_seq = hparams["log_softmax"](pred)

            hyps, _ = hparams["valid_search"](enc_out.detach(), wav_lens)

            ids = batch.id
            tokens_eos, tokens_eos_lens = batch.tokens_eos
            tokens, tokens_lens = batch.tokens

            loss_seq = hparams["seq_cost"](p_seq, tokens_eos, length=tokens_eos_lens).sum()
            loss_ctc = hparams["ctc_cost"](p_ctc, tokens, wav_lens, tokens_lens).sum()

            loss = (hparams["ctc_weight"] * loss_ctc + (1 - hparams["ctc_weight"]) * loss_seq)
            predicted_words = [tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in hyps]
            target_words = [wrd.split(" ") for wrd in batch.wrd]
            wer_metric.append(ids, predicted_words, target_words)
            acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)

            eval_loss = loss.detach().cpu()
            avg_valid_loss = update_average(eval_loss, avg_valid_loss, step)
            logger.info(f"epoch: {epoch}, step: {step}|{total_step}, time: {(time.time()-step_start_time):.2f}s, loss: {eval_loss}, avg_loss: {avg_valid_loss:.4f}")

            if step == 10:
                break
        acc = acc_metric.summarize()
        wer = wer_metric.summarize("error_rate")
        logger.info(f"epoch: {epoch}, time: {time.time()-eval_start_time}, wer: {wer}, acc: {acc}, avg_loss: {avg_valid_loss}")

def train(model, optimizer, train_set, valid_set, tokenizer, hparams):
    for epoch in hparams["epoch_counter"]:
        train_epoch(model, optimizer, train_set, epoch, hparams)
        evaluate(model, valid_set, epoch, hparams, tokenizer)
        # checkpointer.save_and_keep_only(
        #         meta={"ACC": 1.1, "epoch": epoch},
        #         max_keys=["ACC"],
        #         num_to_keep=1,
        #     )


if __name__ == "__main__":
    hparams_file, run_opts, overrides = parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    ddp_init_group(run_opts)
    init_log(run_opts["distributed_launch"])

    # Create experiment directory
    create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    tokenizer = sp.SentencePieceProcessor()
    train_data, valid_data, test_datasets = dataio_prepare(hparams, tokenizer)

    # load LM and tokenizer
    # load_torch_model(hparams["lm_model"], hparams["lm_model_path"], run_opts["device"])
    load_spm(tokenizer, hparams["tokenizer_path"])

    modules = {}
    cnn = ConvolutionFrontEnd(
        input_shape = (8, 10, 80),
        num_blocks = 3,
        num_layers_per_block = 1,
        out_channels = (64, 64, 64),
        kernel_sizes = (5, 5, 1),
        strides = (2, 2, 1),
        residuals = (False, False, True)
    )
    transformer = TransformerASR(
        input_size = 1280,
        tgt_vocab = hparams["output_neurons"],
        d_model = hparams["d_model"],
        nhead = hparams["nhead"],
        num_encoder_layers = hparams["num_encoder_layers"],
        num_decoder_layers = hparams["num_decoder_layers"],
        d_ffn = hparams["d_ffn"],
        dropout = hparams["transformer_dropout"],
        activation = hparams["activation"],
        attention_type = "regularMHA",
        normalize_before = True,
        causal = False
    )
    lm_model = TransformerLM(
        vocab=hparams["output_neurons"], 
        d_model=768,
        nhead=12, 
        num_encoder_layers=12, 
        num_decoder_layers=0, 
        d_ffn=3072, 
        dropout=0.0, 
        activation=torch.nn.GELU,
        normalize_before=False)

    ctc_lin = Linear(input_size=hparams["d_model"], n_neurons=hparams["output_neurons"])
    seq_lin = Linear(input_size=hparams["d_model"], n_neurons=hparams["output_neurons"])
    normalize = InputNormalization(norm_type="global", update_until_epoch=4)
    modules["CNN"] = cnn
    modules["Transformer"] = transformer
    modules["seq_lin"] = seq_lin
    modules["ctc_lin"] = ctc_lin
    modules["normalize"] = normalize

    # modules = torch.nn.ModuleDict(hparams["modules"])
    # tokenizer = hparams["tokenizer"]
    # checkpointer = hparams["checkpointer"]
    model = torch.nn.ModuleDict(modules)
    mm = torch.nn.ModuleList([cnn, transformer, seq_lin, ctc_lin])

    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]
    train_dataloader = make_dataloader(train_data, 'train', run_opts["distributed_launch"], **hparams["train_dataloader_opts"])   # remove checkpoint with dataloader
    valid_dataloader = make_dataloader(valid_data, 'valid', run_opts["distributed_launch"], **hparams["valid_dataloader_opts"])

    # optimizer = hparams["Adam"](modules.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr_adam"], betas=(0.9, 0.98), eps=0.000000001)

    train(model, optimizer, train_dataloader, valid_dataloader, tokenizer, hparams)
