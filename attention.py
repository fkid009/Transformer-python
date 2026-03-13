import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism from "Attention Is All You Need".

    Supports self-attention, cross-attention, and causal (masked) attention
    depending on the inputs and mask provided to forward().

    Usage:
        - Self-attention:  forward(x, x, x)
        - Cross-attention: forward(decoder_out, encoder_out, encoder_out)
        - Causal:          forward(x, x, x, mask=causal_mask)

    Args:
        d_model: Dimension of the model.
        n_heads: Number of attention heads. d_model must be divisible by n_heads.
    """

    def __init__(self, d_model: int, n_heads: int):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            q: Query tensor of shape (batch_size, seq_len_q, d_model).
            k: Key tensor of shape (batch_size, seq_len_k, d_model).
            v: Value tensor of shape (batch_size, seq_len_k, d_model).
            mask: Optional mask of shape (batch_size, 1, 1, seq_len_k) or
                  (batch_size, 1, seq_len_q, seq_len_k). Positions with 0 are masked out.

        Returns:
            Output tensor of shape (batch_size, seq_len_q, d_model).
        """
        batch_size = q.size(0)

        # Linear projections and split into multiple heads
        q = self.Q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) # (batch_size, n_heads, seq_len_q, d_k)
        k = self.K(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) # (batch_size, n_heads, seq_len_q, d_k)
        v = self.V(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) # (batch_size, n_heads, seq_len_q, d_k)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)                # (batch_size, n_heads, seq_len_q, seq_len_k)
        attn_output = torch.matmul(attn_weights, v)                 # (batch_size, n_heads, seq_len_q, d_k)

        # Concatenate heads and pass through output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.output_linear(attn_output) # (batch_size, n_heads, seq_len_q, d_k)
        return output
