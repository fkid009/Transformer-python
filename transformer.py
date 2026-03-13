import torch
import torch.nn as nn

from embedding import TransformerEmbedding
from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):
    """Transformer model for sequence-to-sequence tasks.

    Args:
        src_vocab_size: Source vocabulary size.
        tgt_vocab_size: Target vocabulary size.
        d_model: Dimension of the model.
        n_heads: Number of attention heads.
        n_layers: Number of encoder/decoder layers.
        max_len: Maximum sequence length.
        pad_idx: Padding token index.
        dropout: Dropout rate.
    """

    def __init__(
            self,
            src_vocab_size: int,
            tgt_vocab_size: int,
            d_model: int,
            n_heads: int,
            n_layers: int,
            max_len: int,
            pad_idx: int = 0,
            dropout: float = 0.1
    ):
        super().__init__()

        self.pad_idx = pad_idx

        self.src_embedding = TransformerEmbedding(src_vocab_size, d_model, max_len, pad_idx)
        self.tgt_embedding = TransformerEmbedding(tgt_vocab_size, d_model, max_len, pad_idx)

        self.encoder = Encoder(d_model, n_heads, n_layers, dropout)
        self.decoder = Decoder(d_model, n_heads, n_layers, dropout)

        self.output_linear = nn.Linear(d_model, tgt_vocab_size)

    def forward(
            self,
            src: torch.Tensor,
            tgt: torch.Tensor,
            src_mask: torch.Tensor = None,
            tgt_mask: torch.Tensor = None,
            cross_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            src: Source token IDs of shape (batch_size, src_seq_len).
            tgt: Target token IDs of shape (batch_size, tgt_seq_len).
            src_mask: Optional source padding mask.
            tgt_mask: Optional target causal + padding mask.
            cross_mask: Optional cross-attention padding mask.

        Returns:
            Logits of shape (batch_size, tgt_seq_len, tgt_vocab_size).
        """
        # Mask 생성
        if src_mask is None:
            src_mask = self._pad_mask(src)
        if tgt_mask is None:
            tgt_pad_mask = self._pad_mask(tgt)
            causal_mask = self._causal_mask(tgt.size(1)).to(tgt.device)
            tgt_mask = tgt_pad_mask & causal_mask
        if cross_mask is None:
            cross_mask = self._pad_mask(src)

        # Encode
        enc_out = self.encoder(self.src_embedding(src), src_mask)

        # Decode
        dec_out = self.decoder(self.tgt_embedding(tgt), enc_out, tgt_mask, cross_mask)

        return self.output_linear(dec_out)

    def _pad_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """Create padding mask: 1 for real tokens, 0 for padding."""
        return (seq != self.pad_idx).unsqueeze(1).unsqueeze(2).to(seq.device)  # (batch_size, 1, 1, seq_len)

    def _causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create causal (look-ahead) mask: lower triangular = 1, upper = 0."""
        return torch.tril(torch.ones(seq_len, seq_len)).bool().unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, seq_len)
