import torch
import torch.nn as nn

from attention import MultiHeadAttention

class DecoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            dropout: float = 0.1
    ):
        super(DecoderLayer, self).__init__()

        self.dff = 4 * d_model  

        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, self.dff),
            nn.ReLU(),
            nn.Linear(self.dff, d_model)
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
            cross_mask: torch.Tensor = None):
        
        # Self-attention
        attn_output = self.self_attn(x, x, x, self_mask)  # (batch_size, seq_len, d_model)
        x = self.norm1(x + self.dropout1(attn_output))  # Residual connection + LayerNorm

        # Cross-attention
        attn_output = self.cross_attn(x, enc_out, enc_out, cross_mask)  # (batch_size, seq_len, d_model)
        x = self.norm2(x + self.dropout2(attn_output))  # Residual connection + LayerNorm

        # Feed-forward network
        ffn_output = self.ffn(x)  # (batch_size, seq_len, d_model)
        x = self.norm3(x + self.dropout3(ffn_output))  # Residual connection + LayerNorm
        return x
    

class Decoder(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_heads: int,
            n_layers: int,
            dropout: float = 0.1
    ):
        super(Decoder, self).__init__()

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
        
        for layer in self.layers:
            x = layer(x, enc_out, self_mask, cross_mask)
        return x