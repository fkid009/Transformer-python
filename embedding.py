import torch
import torch.nn as nn
import math


class TransformerEmbedding(nn.Module):
    """Token Embedding + Positional Encoding for Transformer.

    Combines learnable token embeddings with fixed sinusoidal positional encodings.
    Token embeddings are scaled by sqrt(d_model) as described in "Attention Is All You Need".

    Args:
        vocab_size: Size of the vocabulary.
        d_model: Dimension of the embedding vectors.
        max_len: Maximum sequence length for positional encoding.
        pad_idx: Index of the padding token (default: 0).
    """

    def __init__(
            self, 
            vocab_size: int,
            d_model: int, 
            max_len: int, 
            pad_idx: int = 0
    ):
        super(TransformerEmbedding, self).__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.d_model = d_model

        # Positional Encoding (fixed, not learnable)
        # PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        pe = torch.zeros(max_len, d_model)                          # (max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)                 # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )                                                            # (d_model/2,)

        pe[:, 0::2] = torch.sin(pos * div_term)  # even indices
        pe[:, 1::2] = torch.cos(pos * div_term)  # odd indices

        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token indices of shape (batch_size, seq_len).

        Returns:
            Embedded tensor of shape (batch_size, seq_len, d_model).
        """
        seq_len = x.size(1)

        # Scale token embeddings and add positional encoding
        x = self.token_embedding(x) * math.sqrt(self.d_model)  # (batch_size, seq_len, d_model)
        x = x + self.pe[:, :seq_len, :]                        # (batch_size, seq_len, d_model)

        return x
