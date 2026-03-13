import torch
import torch.nn as nn

from attention import MultiHeadAttention


class DecoderLayer(nn.Module):
    """Single decoder layer: Masked Self-Attn → Cross-Attn → FFN, each with Add & Norm.

    Args:
        d_model: Dimension of the model.
        n_heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()

        self.d_ff = 4 * d_model

        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, self.d_ff),
            nn.ReLU(),
            nn.Linear(self.d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            enc_out: torch.Tensor,
            self_mask: torch.Tensor = None,
            cross_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: Decoder input of shape (batch_size, tgt_seq_len, d_model).
            enc_out: Encoder output of shape (batch_size, src_seq_len, d_model).
            self_mask: Causal + pad mask of shape (batch_size, 1, tgt_seq_len, tgt_seq_len).
            cross_mask: Padding mask of shape (batch_size, 1, 1, src_seq_len).

        Returns:
            Output tensor of shape (batch_size, tgt_seq_len, d_model).
        """
        # Masked self-attention
        attn_output = self.self_attn(x, x, x, self_mask)       # (batch_size, tgt_seq_len, d_model)
        x = self.norm1(x + self.dropout1(attn_output))

        # Cross-attention
        attn_output = self.cross_attn(x, enc_out, enc_out, cross_mask)  # (batch_size, tgt_seq_len, d_model)
        x = self.norm2(x + self.dropout2(attn_output))

        # Feed-forward network
        ffn_output = self.ffn(x)                                # (batch_size, tgt_seq_len, d_model)
        x = self.norm3(x + self.dropout3(ffn_output))

        return x


class Decoder(nn.Module):
    """Transformer decoder: stack of N DecoderLayers.

    Args:
        d_model: Dimension of the model.
        n_heads: Number of attention heads.
        n_layers: Number of decoder layers.
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float = 0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])

    def forward(
            self,
            x: torch.Tensor,
            enc_out: torch.Tensor,
            self_mask: torch.Tensor = None,
            cross_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: Decoder input of shape (batch_size, tgt_seq_len, d_model).
            enc_out: Encoder output of shape (batch_size, src_seq_len, d_model).
            self_mask: Causal + pad mask for decoder self-attention.
            cross_mask: Padding mask for cross-attention.

        Returns:
            Output tensor of shape (batch_size, tgt_seq_len, d_model).
        """
        for layer in self.layers:
            x = layer(x, enc_out, self_mask, cross_mask)
        return x
