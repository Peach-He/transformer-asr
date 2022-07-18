import math
import torch
import logging
import torch.nn as nn

logger = logging.getLogger(__name__)


class Embedding(nn.Module):

    def __init__(
        self,
        num_embeddings,
        embedding_dim=128,
        consider_as_one_hot=False,
        blank_id=0,
    ):

        super().__init__()
        self.num_embeddings = num_embeddings
        self.consider_as_one_hot = consider_as_one_hot
        if self.consider_as_one_hot:
            self.embedding_dim = self.num_embeddings - 1
        else:
            self.embedding_dim = embedding_dim
        self.blank_id = blank_id

        if self.consider_as_one_hot:
            # deal with blank_id, the output should be embedding_dim-1 as we consider blank output as zeros one_hot vect
            # padding_idx fix the idx row to zeros
            self.Embedding = nn.Embedding(
                self.num_embeddings,
                self.embedding_dim,
                padding_idx=self.blank_id,
            )
            one_hot = torch.eye(self.embedding_dim)
            if self.blank_id + 1 != self.num_embeddings:
                self.Embedding.weight.data[self.blank_id + 1 :] = one_hot[
                    self.blank_id :
                ]
            if self.blank_id != 0:
                self.Embedding.weight.data[: self.blank_id] = one_hot[
                    : self.blank_id
                ]
            self.Embedding.weight.requires_grad = False
        else:
            self.Embedding = nn.Embedding(
                self.num_embeddings, self.embedding_dim
            )

    def forward(self, x):
        """Returns the embedding of input tensor.

        Arguments
        ---------
        x : torch.Tensor
           Input to embed.
        """
        # pytorch embedding layer only accept long dtype
        return self.Embedding(x.long())


class NormalizedEmbedding(nn.Module):
    """This class implements the normalized embedding layer for the transformer.

    Since the dot product of the self-attention is always normalized by sqrt(d_model)
    and the final linear projection for prediction shares weight with the embedding layer,
    we multiply the output of the embedding by sqrt(d_model).

    Arguments
    ---------
    d_model: int
        The number of expected features in the encoder/decoder inputs (default=512).
    vocab: int
        The vocab size.

    Example
    -------
    >>> emb = NormalizedEmbedding(512, 1000)
    >>> trg = torch.randint(0, 999, (8, 50))
    >>> emb_fea = emb(trg)
    """

    def __init__(self, d_model, vocab):
        super().__init__()
        self.emb = Embedding(
            num_embeddings=vocab, embedding_dim=d_model, blank_id=0
        )
        self.d_model = d_model

    def forward(self, x):
        """ Processes the input tensor x and returns an output tensor."""
        return self.emb(x) * math.sqrt(self.d_model)