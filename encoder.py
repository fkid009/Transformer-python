import torch
import torch.nn as nn

from attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    """Single encoder layer: Self-Attention → Add & Norm → FFN → Add & Norm.

    Args:
        d_model: Dimension of the model.
        n_heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()

        self.d_ff = 4 * d_model

        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, self.d_ff),
            nn.ReLU(),
            nn.Linear(self.d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            mask: Optional padding mask of shape (batch_size, 1, 1, seq_len).

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Self-attention
        attn_output = self.self_attn(x, x, x, mask)            # (batch_size, seq_len, d_model)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed-forward network
        ffn_output = self.ffn(x)                                # (batch_size, seq_len, d_model)
        x = self.norm2(x + self.dropout2(ffn_output))

        return x


class Encoder(nn.Module):
    """Transformer encoder: stack of N EncoderLayers.

    Args:
        d_model: Dimension of the model.
        n_heads: Number of attention heads.
        n_layers: Number of encoder layers.
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float = 0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            mask: Optional padding mask of shape (batch_size, 1, 1, seq_len).

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x
