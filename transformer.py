import torch
import torch.nn as nn

from embedding import TransformerEmbedding
from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):
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
        super(Transformer, self).__init__()

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
        # Mask 생성
        if src_mask is None:
            src_mask = self._pad_mask(src, self.src_embedding.token_embedding.padding_idx) if src is not None else None
        if tgt_mask is None:
            tgt_pad_mask = self._pad_mask(tgt, self.tgt_embedding.token_embedding.padding_idx) if tgt is not None else None
            causal_mask = self._causal_mask(tgt.size(1)).to(tgt.device)
            tgt_mask = tgt_pad_mask & causal_mask  # pad + causal 결합
        if cross_mask is None:
            cross_mask = self._pad_mask(src, self.src_embedding.token_embedding.padding_idx) if src is not None else None

        # Encode
        enc_out = self.encoder(self.src_embedding(src), src_mask)           # (batch_size, src_seq_len, d_model)

        # Decode
        dec_out = self.decoder(self.tgt_embedding(tgt), enc_out, tgt_mask, cross_mask)  # (batch_size, tgt_seq_len, d_model)
        output = self.output_linear(dec_out)  # (batch_size, tgt_seq_len, tgt_vocab_size)
        return output
    
    def _pad_mask(self, seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
    
    def _causal_mask(self, seq_len: int) -> torch.Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len)).bool()  # (seq_len, seq_len)
        return mask.unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, seq_len)