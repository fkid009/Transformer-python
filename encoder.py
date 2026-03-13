import torch
import torch.nn as nn

from attention import MultiHeadAttention

class EncoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            dropout: float = 0.1
    ):
        super(EncoderLayer, self).__init__()

        self.dff = 4 * d_model  

        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, self.dff),
            nn.ReLU(),
            nn.Linear(self.dff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention
        attn_output = self.self_attn(x, x, x, mask)  # (batch_size, seq_len, d_model)
        x = self.norm1(x + self.dropout1(attn_output))  # Residual connection + LayerNorm

        # Feed-forward network
        ffn_output = self.ffn(x)  # (batch_size, seq_len, d_model)
        x = self.norm2(x + self.dropout2(ffn_output))  # Residual connection + LayerNorm
        return x
    

class Encoder(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            n_layers: int,
            dropout: float = 0.1
    ):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x