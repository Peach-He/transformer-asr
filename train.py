import sys
import torch
import logging
import time
import yaml
import sentencepiece as sp

from utils.distributed import run_on_main, ddp_init_group
from data.dataio.dataloader import make_dataloader
from data.dataio.dataset import dataio_prepare
from utils.utils import check_gradients, update_average, create_experiment_directory, init_log, parse_args
from utils.parameter_transfer import load_torch_model, load_spm
from model.module.convolution import ConvolutionFrontEnd
from model.TransformerASR import TransformerASR
from model.module.linear import Linear
from data.processing.features import InputNormalization
from utils.checkpoints import Checkpointer
from model.TransformerLM import TransformerLM
from model.decoders.seq2seq import S2STransformerBeamSearch, batch_filter_seq2seq_output
from trainer.losses import ctc_loss, kldiv_loss
from trainer.schedulers import NoamScheduler
from data.augment import SpecAugment
from data.features import Fbank
from utils.Accuracy import AccuracyStats
from utils.metric_stats import ErrorRateStats


def train_epoch(model, optimizer, train_set, epoch, hparams, scheduler, feat_proc):
    logger = logging.getLogger("train")

    augment = SpecAugment(**hparams["augmentation"])
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
        feats = feat_proc(wavs)
        feats = model["normalize"](feats, wav_lens, epoch=epoch)
        feats = augment(feats)

        src = model["CNN"](feats)
        enc_out, pred = model["Transformer"](src, tokens_bos, wav_lens, pad_idx=hparams["pad_index"])

        logits = model["ctc_lin"](enc_out)
        p_ctc = logits.log_softmax(dim=-1)

        pred = model["seq_lin"](pred)
        p_seq = pred.log_softmax(dim=-1)

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        loss_seq = kldiv_loss(p_seq, tokens_eos, length=tokens_eos_lens, label_smoothing=hparams["label_smoothing"], reduction=hparams["loss_reduction"]).sum()
        loss_ctc = ctc_loss(p_ctc, tokens, wav_lens, tokens_lens, blank_index=hparams["blank_index"], reduction=hparams["loss_reduction"]).sum()

        loss = (hparams["ctc_weight"] * loss_ctc + (1 - hparams["ctc_weight"]) * loss_seq)
        (loss / hparams["grad_accumulation_factor"]).backward()

        if should_step:
            is_loss_finite, nonfinite_count = check_gradients(model, loss, hparams["max_grad_norm"], nonfinite_count)
            if is_loss_finite:
                optimizer.step()
            optimizer.zero_grad()
            scheduler(optimizer)

        train_loss = loss.detach().cpu()
        avg_train_loss = update_average(train_loss, avg_train_loss, step)
        logger.info(f"epoch: {epoch}, step: {step}|{total_step}, time: {(time.time()-step_start_time):.2f}s, loss: {train_loss}, avg_loss: {avg_train_loss:.4f}, lr: {scheduler.current_lr}")

    logger.info(f"epoch: {epoch}, time: {(time.time()-epoch_start_time):.2f}s, avg_loss: {avg_train_loss:.4f}")


def evaluate(model, valid_set, epoch, hparams, tokenizer, searcher, feat_proc):
    logger = logging.getLogger("evaluate")

    acc_metric = AccuracyStats()
    wer_metric = ErrorRateStats()
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
            feats = feat_proc(wavs)
            feats = model["normalize"](feats, wav_lens, epoch=epoch)

            src = model["CNN"](feats)
            enc_out, pred = model["Transformer"](src, tokens_bos, wav_lens, pad_idx=hparams["pad_index"])

            logits = model["ctc_lin"](enc_out)
            p_ctc = logits.log_softmax(dim=-1)

            pred = model["seq_lin"](pred)
            p_seq = pred.log_softmax(dim=-1)

            # hyps, _ = searcher(enc_out.detach(), wav_lens)
            # v, k = p_seq.max(0)
            # hyps = batch_filter_seq2seq_output(k, hparams["eos_index"])

            ids = batch.id
            tokens_eos, tokens_eos_lens = batch.tokens_eos
            tokens, tokens_lens = batch.tokens

            loss_seq = kldiv_loss(p_seq, tokens_eos, length=tokens_eos_lens, label_smoothing=hparams["label_smoothing"], reduction=hparams["loss_reduction"]).sum()
            loss_ctc = ctc_loss(p_ctc, tokens, wav_lens, tokens_lens, blank_index=hparams["blank_index"], reduction=hparams["loss_reduction"]).sum()

            loss = (hparams["ctc_weight"] * loss_ctc + (1 - hparams["ctc_weight"]) * loss_seq)
            predicted_words = [tokenizer.decode_ids(utt_seq.tolist()).split(" ") for utt_seq in p_seq.argmax(-1)]
            target_words = [wrd.split(" ") for wrd in batch.wrd]
            wer_metric.append(ids, predicted_words, target_words)
            acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)

            eval_loss = loss.detach().cpu()
            avg_valid_loss = update_average(eval_loss, avg_valid_loss, step)
            # logger.info(f"epoch: {epoch}, step: {step}|{total_step}, time: {(time.time()-step_start_time):.2f}s, loss: {eval_loss}, avg_loss: {avg_valid_loss:.4f}")

        acc = acc_metric.summarize()
        wer = wer_metric.summarize("error_rate")
        logger.info(f"epoch: {epoch}, time: {time.time()-eval_start_time}, wer: {wer}, acc: {acc}, avg_loss: {avg_valid_loss}")

def train(model, optimizer, train_set, valid_set, searcher, tokenizer, hparams):
    scheduler = NoamScheduler(lr_initial=hparams["lr_adam"], n_warmup_steps=hparams["n_warmup_steps"])
    feat_proc = Fbank(**hparams["compute_features"])

    for epoch in range(1, hparams["epochs"]+1):
        train_epoch(model, optimizer, train_set, epoch, hparams, scheduler, feat_proc)
        evaluate(model, valid_set, epoch, hparams, tokenizer, searcher, feat_proc)
        # checkpointer.save_and_keep_only(
        #         meta={"ACC": 1.1, "epoch": epoch},
        #         max_keys=["ACC"],
        #         num_to_keep=1,
        #     )


if __name__ == "__main__":
    args = parse_args()
    with open(args.param_file, 'r') as f:
        hparams = yaml.safe_load(f)
    
    torch.manual_seed(args.seed)

    ddp_init_group(args)
    init_log(args.distributed_launch)

    # Create experiment directory
    create_experiment_directory(args.output_folder)

    tokenizer = sp.SentencePieceProcessor()
    train_data, valid_data, test_datasets = dataio_prepare(args, hparams, tokenizer)

    # load tokenizer
    load_spm(tokenizer, args.tokenizer_ckpt)

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
        activation = torch.nn.GELU,
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

    # checkpointer = hparams["checkpointer"]
    model = torch.nn.ModuleDict(modules)
    mm = torch.nn.ModuleList([cnn, transformer, seq_lin, ctc_lin])

    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]
    train_dataloader = make_dataloader(train_data, 'train', args.distributed_launch, **hparams["train_dataloader_opts"])   # remove checkpoint with dataloader
    valid_dataloader = make_dataloader(valid_data, 'valid', args.distributed_launch, **hparams["valid_dataloader_opts"])

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr_adam"], betas=(0.9, 0.98), eps=0.000000001)

    valid_searcher = S2STransformerBeamSearch(
        modules = [transformer, seq_lin, ctc_lin],
        bos_index = hparams["bos_index"], 
        eos_index = hparams["eos_index"], 
        blank_index = hparams["blank_index"],
        min_decode_ratio = hparams["min_decode_ratio"],
        max_decode_ratio = hparams["max_decode_ratio"], 
        beam_size = hparams["valid_beam_size"], 
        ctc_weight = hparams["ctc_weight_decode"], 
        using_eos_threshold = False,
        length_normalization = False
    )

    train(model, optimizer, train_dataloader, valid_dataloader, valid_searcher, tokenizer, hparams)
