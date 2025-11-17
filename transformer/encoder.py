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
    
class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len).unsqueeze(1)

        dim_indices = torch.arange(0, emb_dim, 2).float()

        div_term = 1.0 / (10000.0 ** (dim_indices / emb_dim)) # torch.exp(dim_indices * (-math.log(10000.0) / emb_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, emb_dim)

        self.register_buffer('pe', pe) # non-trainable buffer

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
    
class PointwiseFeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.activation = nn.ReLU() # nn.GELU()
        self.dropout = nn.Dropout(dropout)
        # self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # residual = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        # x = self.norm(residual + x) # residual connection with add & norm
        return x
    

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.pff = PointwiseFeedForward(hidden_size, intermediate_size, dropout)
        self.pff_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, pad_mask = None):
        mha_output = self.mha(x, pad_mask)
        mha_output = self.dropout(mha_output)
        mha_output = self.attn_norm(x + mha_output)  # Add & Norm

        x = self.pff(mha_output)
        x = self.pff_norm(mha_output + x)  # Add & Norm  
        return x
        
class Encoder(nn.Module):
    def __init__(self, num_layers, hidden_size, intermediate_size, num_heads, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(hidden_size, intermediate_size, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, pad_mask = None):
        for layer in self.layers:
            x = layer(x, pad_mask)
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_encoder = PositionalEncoding(config.hidden_size, config.max_len)
        self.pos_dropout = nn.Dropout(config.dropout)
        self.encoder = Encoder(
            num_layers = config.num_layers,
            hidden_size = config.hidden_size,
            num_heads = config.num_heads,
            intermediate_size = config.intermediate_size,
            dropout = config.dropout
        )
    
    def forward(self, input_ids, pad_mask = None):
        x = self.embedding(input_ids)
        x = self.pos_encoder(x)
        x = self.pos_dropout(x)
        x = self.encoder(x, pad_mask)
        return x