import torch
from torch import nn

from model.module.linear import Linear
from model.module.normalization import LayerNorm
from model.module.embedding import NormalizedEmbedding
from model.module.positional_encoding import PositionalEncoding
from model.module.attention import RelPosEncXL
from model.module.containers import ModuleList
from model.transformer_encoder import TransformerEncoder
from model.transformer_decoder import TransformerDecoder
from model.utils import length_to_mask, get_key_padding_mask, get_lookahead_mask



class TransformerLM(nn.Module):
    """This is an implementation of transformer language model.

    The architecture is based on the paper "Attention Is All You Need": https://arxiv.org/pdf/1706.03762.pdf

    Arguments
    ----------
    d_model : int
        The number of expected features in the encoder/decoder inputs (default=512).
    nhead : int
        The number of heads in the multiheadattention models (default=8).
    num_encoder_layers : int
        The number of sub-encoder-layers in the encoder (default=6).
    num_decoder_layers : int
        The number of sub-decoder-layers in the decoder (default=6).
    dim_ffn : int
        The dimension of the feedforward network model (default=2048).
    dropout : int
        The dropout value (default=0.1).
    activation: torch class
        The activation function of encoder/decoder intermediate layer, relu or gelu (default=relu).
    decoder_use_memory: bool
        whether to use the hidden state in the decoder

    Example
    -------
    >>> src = torch.randint(0, 720, [8, 120])
    >>> net = TransformerLM(720, 512, 8, 1, 0, 1024, activation=torch.nn.GELU)
    >>> enc_out = net.forward(src)
    >>> print(enc_out.shape)
    torch.Size([8, 120, 720])
    """

    def __init__(
        self,
        vocab,
        d_model=512,
        nhead=8,
        num_encoder_layers=12,
        num_decoder_layers=0,
        d_ffn=2048,
        dropout=0.1,
        activation=nn.ReLU,
        positional_encoding="fixed_abs_sine",
        normalize_before=False,
        d_embedding=None,
        max_length=2500,
        causal=True,
        attention_type="regularMHA",
        decoder_use_memory=False,
    ):
        super().__init__()
        self.causal = causal
        self.attention_type = attention_type
        self.positional_encoding_type = positional_encoding
        self.encoder_kdim = None
        self.encoder_vdim = None
        self.decoder_kdim = None
        self.decoder_vdim = None

        if positional_encoding == "fixed_abs_sine":
            self.positional_encoding = PositionalEncoding(d_model, max_length)
        elif positional_encoding is None:
            pass

        # overrides any other pos_embedding
        if attention_type == "RelPosMHAXL":
            self.positional_encoding = RelPosEncXL(d_model)
            self.positional_encoding_decoder = PositionalEncoding(
                d_model, max_length
            )

        # initialize the encoder
        if num_encoder_layers > 0:
            self.encoder = TransformerEncoder(
                nhead=nhead,
                num_layers=num_encoder_layers,
                d_ffn=d_ffn,
                d_model=d_model,
                dropout=dropout,
                activation=activation,
                normalize_before=normalize_before,
                causal=self.causal,
                attention_type=self.attention_type,
                kdim=self.encoder_kdim,
                vdim=self.encoder_vdim,
            )

        # initialize the decoder
        if num_decoder_layers > 0:
            self.decoder = TransformerDecoder(
                num_layers=num_decoder_layers,
                nhead=nhead,
                d_ffn=d_ffn,
                d_model=d_model,
                dropout=dropout,
                activation=activation,
                normalize_before=normalize_before,
                causal=True,
                attention_type="regularMHA",  # always use regular attention in decoder
                kdim=self.decoder_kdim,
                vdim=self.decoder_vdim,
            )
            
        self.d_embedding = d_embedding
        if d_embedding is None:
            self.d_embedding = d_model

        self.custom_src_module = NormalizedEmbedding(self.d_embedding, vocab)

        self.embedding_proj = None
        if d_embedding is not None:
            self.embedding_proj = Linear(
                input_size=self.d_embedding, n_neurons=d_model
            )

        self.output_proj = ModuleList(
            Linear(input_size=d_model, n_neurons=d_model),
            LayerNorm(d_model, eps=1e-6),
            Linear(input_size=d_model, n_neurons=vocab),
        )

        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.decoder_use_memory = decoder_use_memory

        # reset the params of the transformer model
        self._reset_params()

    def forward(self, src, hx=None):
        """
        Arguments
        ---------
        src : tensor
            The sequence to the encoder (required).
        """
        src_mask, src_key_padding_mask = self.make_masks(src)

        src = self.custom_src_module(src)
        if self.embedding_proj is not None:
            src = self.embedding_proj(src)
        src = src + self.positional_encoding(src)
        if self.num_encoder_layers > 0:
            encoder_out, _ = self.encoder(
                src=src,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
            )

        if self.num_decoder_layers > 0:
            if self.decoder_use_memory:
                encoder_out, _, _ = self.decoder(
                    tgt=src,
                    memory=encoder_out,
                    tgt_mask=src_mask,
                    tgt_key_padding_mask=src_key_padding_mask,
                )
            else:
                encoder_out, _ = self.decoder(
                    src=src,
                    tgt=src,
                    tgt_mask=src_mask,
                    tgt_key_padding_mask=src_key_padding_mask,
                )

        pred = self.output_proj(encoder_out)
        return pred

    def _reset_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)

    def make_masks(
        self, src, pad_idx=0, look_ahead_mask=True, padding_mask=True
    ):
        src_mask = None
        if look_ahead_mask:
            src_mask = get_lookahead_mask(src)

        src_key_padding_mask = None
        if padding_mask:
            src_key_padding_mask = get_key_padding_mask(src, pad_idx)

        return src_mask, src_key_padding_mask
