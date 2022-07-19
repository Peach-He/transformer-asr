import os
import time
import torch
from tqdm.contrib import tqdm
import logging
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.parse_args import parse_arguments
from utils.load_hparams import load_hyperpyyaml
import utils.distributed as dist
from utils.utils import run_on_main, is_main_process, check_gradients, update_average
from model.transformer import Transformer
from data.dataloader import dataio_prepare, make_dataloader


logger = logging.getLogger(__name__)

def train_epoch(model, optimizer, train_set, epoch, hparams, progressbar):
    model.train()
    
    step = 0
    avg_train_loss = 0.0
    epoch_start_time = time.time()
    with tqdm(train_set, initial=0, dynamic_ncols=True, disable=not progressbar) as t:
        for batch in t:
            step += 1
            step_start_time = time.time()
            should_step = step % hparams["grad_accumulation_factor"] == 0
            wavs, wav_lens = batch.sig
            tokens_bos, _ = batch.tokens_bos
            feats = hparams["compute_features"](wavs)
            feats = hparams["modules"]["normalize"](feats, wav_lens, epoch=epoch)
            feats = hparams["augmentation"](feats)

            src = hparams["modules"]["CNN"](feats)
            enc_out, pred = hparams["modules"]["Transformer"](src, tokens_bos, wav_lens, pad_idx=hparams["pad_index"])

            logits = hparams["modules"]["ctc_lin"](enc_out)
            p_ctc = hparams["log_softmax"](logits)

            pred = hparams["modules"]["seq_lin"](pred)
            p_seq = hparams["log_softmax"](pred)

            ids = batch.id
            tokens_eos, tokens_eos_lens = batch.tokens_eos
            tokens, tokens_lens = batch.tokens

            loss_seq = hparams["seq_cost"](p_seq, tokens_eos, length=tokens_eos_lens).sum()
            loss_ctc = hparams["ctc_cost"](p_ctc, tokens, wav_lens, tokens_lens).sum()
            loss = (hparams["ctc_weight"] * loss_ctc + (1 - hparams["ctc_weight"]) * loss_seq)
            loss = loss / hparams["grad_accumulation_factor"]
            loss.backward()

            if should_step:
                if check_gradients(loss):
                    torch.nn.utils.clip_grad_norm_((p for p in model.parameters()), hparams["max_grad_norm"])
                    optimizer.step()
                optimizer.zero_grad()

            train_loss = loss.detach().cpu()
            avg_train_loss = update_average(train_loss, avg_train_loss, step)
            t.set_postfix(train_loss=avg_train_loss)
            logger.info(f"epoch: {epoch}, step: {step}, time: {time.time()-step_start_time}, loss: {train_loss}, lr: {hparams['noam_annealing'].current_lr}")
    logger.info(f"epoch: {epoch}, time: {time.time()-epoch_start_time}, avg_loss: {avg_train_loss}")



def evaluate(model, valid_set, epoch, hparams, tokenizer, progressbar):
    acc_metric = hparams["acc_computer"]()
    wer_metric = hparams["error_rate_computer"]()
    model.eval()
    avg_valid_loss = 0.0
    step = 0
    eval_start_time = time.time()
    with torch.no_grad():
        for batch in tqdm(valid_set, dynamic_ncols=True, disable=not progressbar):
            step += 1
            wavs, wav_lens = batch.sig
            tokens_bos, _ = batch.tokens_bos
            feats = hparams["compute_features"](wavs)
            feats = hparams["modules"]["normalize"](feats, wav_lens, epoch=epoch)

            src = hparams["modules"]["CNN"](feats)
            enc_out, pred = hparams["modules"]["Transformer"](src, tokens_bos, wav_lens, pad_idx=hparams["pad_index"])

            logits = hparams["modules"]["ctc_lin"](enc_out)
            p_ctc = hparams["log_softmax"](logits)

            pred = hparams["modules"]["seq_lin"](pred)
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

        acc = acc_metric.summarize()
        wer = wer_metric.summarize("error_rate")
        logger.info(f"epoch: {epoch}, time: {time.time()-eval_start_time}, wer: {wer}, acc: {acc}, avg_loss: {avg_valid_loss}")

def train(model, optimizer, train_set, valid_set, tokenizer, checkpointer, hparams):
    progressbar = is_main_process()

    for epoch in hparams["epoch_counter"]:
        train_epoch(model, optimizer, train_set, epoch, hparams, progressbar)
        evaluate(model, valid_set, epoch, hparams, tokenizer, progressbar)
        checkpointer.save_and_keep_only(
                meta={"ACC": 1.1, "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=1,
            )


def main():
    args = parse_arguments()
    with open(args.param_file) as f:
        hparams = load_hyperpyyaml(f)

    if args.distributed_launch or (int(os.environ.get('WORLD_SIZE', 1)) > 1):
        dist.init_distributed(backend=args.distributed_backend)
        world_size = dist.my_size
    else:
        world_size = 1
    
    train_data, valid_data, test_datasets, tokenizer = dataio_prepare(hparams)
    train_dataloader = make_dataloader(train_data, 'train', world_size>1, **hparams["train_dataloader_opts"])   # remove checkpoint with dataloader
    valid_dataloader = make_dataloader(valid_data, 'valid', world_size>1, **hparams["valid_dataloader_opts"])


    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=args.device)

    tokenizer = hparams["tokenizer"]
    checkpointer = hparams["checkpointer"]

    model = torch.nn.ModuleDict(hparams["modules"])
    # model = Transformer(tgt_vocab=hparams['output_neurons'], input_size=hparams['Transformer']['input_size'], 
    #     d_model=hparams['d_model'], nhead=hparams['nhead'], num_encoder_layers=hparams['num_encoder_layers'], 
    #     num_decoder_layers=hparams['num_decoder_layers'], d_ffn=hparams['d_ffn'], dropout=hparams['transformer_dropout'], 
    #     activation=hparams['activation'], positional_encoding=, normalize_before=hparams['Transformer']['normalize_before'], 
    #     attention_type=hparams['Transformer']['attention_type'], max_length=, causal=hparams['Transformer']['causal'], 
    #     encoder_kdim=, encoder_vdim=, encoder_kdim=, decoder_vdim=)

    if world_size > 1:
        model = DDP(model)
    # print(model)

    optimizer = hparams["Adam"](model.parameters())
    checkpointer.add_recoverable("optimizer", optimizer)

    train(model, optimizer, train_dataloader, valid_dataloader, tokenizer, checkpointer, hparams)


if __name__ == "__main__":
    main()
