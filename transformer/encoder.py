import torch
import torch.nn as nn
import torch.nn.functional as F

# class SelfAttention(nn.Module):
#     def __init__(self,emb_dim, dropout=0.1):
#         super().__init__()

#         self.emb_dim = emb_dim

#         self.linear_q = nn.Linear(emb_dim, emb_dim)
#         self.linear_k = nn.Linear(emb_dim, emb_dim) 
#         self.linear_v = nn.Linear(emb_dim, emb_dim)

#         self.scale = self.emb_dim ** 0.5
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         Q = self.linear_q(x) 
#         K = self.linear_k(x) 
#         V = self.linear_v(x) 

#         scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale # torch.bmm(Q, K.transpose(-2, -1)) / self.scale

#         attn_weights = F.softmax(scores, dim=-1)
#         attn_weights = self.dropout(attn_weights)

#         context = torch.matmul(attn_weights, V)

#         return context, attn_weights
        

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0, "Embedding dimension must be divisible by number of heads."

        self.emb_dim = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)

        self.scale = self.head_size ** 0.5
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, pad_mask = None):
        batch_size, seq_len, _ = x.size()

        Q = self.linear_q(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2) # (bath_size, num_heads, seq_len, head_dim)
        K = self.linear_k(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        V = self.linear_v(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale # (batch_size, num_heads, seq_len, seq_len)

        if pad_mask is not None:
            pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(pad_mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V) # context vector per head

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.emb_dim) # concatenate heads: (batch_size, seq_len, emb_dim)

        output = self.linear_out(context) # final linear transformation

        return output