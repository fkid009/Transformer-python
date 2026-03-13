import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset

from tokenizer import Tokenizer
from transformer import Transformer

import torch


class TranslationDataset(Dataset):
    """Multi30k translation dataset wrapper.

    Args:
        data: List of dicts with 'en' and 'de' keys.
        src_tokenizer: Tokenizer for source language (English).
        tgt_tokenizer: Tokenizer for target language (German).
        max_len: Maximum sequence length.
    """

    def __init__(self, data, src_tokenizer: Tokenizer, tgt_tokenizer: Tokenizer, max_len: int = 128):
        self.data = data
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text = self.data[idx]["en"]
        tgt_text = self.data[idx]["de"]

        src_ids = self.src_tokenizer.encode(src_text)
        tgt_ids = self.tgt_tokenizer.encode(tgt_text)

        src_ids = self.src_tokenizer.pad(src_ids, self.max_len)
        tgt_ids = self.tgt_tokenizer.pad(tgt_ids, self.max_len)

        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "tgt": torch.tensor(tgt_ids, dtype=torch.long),
        }


class TransformerLitModule(pl.LightningModule):
    """PyTorch Lightning module for Transformer training.

    Args:
        src_vocab_size: Source vocabulary size.
        tgt_vocab_size: Target vocabulary size.
        d_model: Model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of encoder/decoder layers.
        max_len: Maximum sequence length.
        pad_idx: Padding token index.
        dropout: Dropout rate.
        lr: Learning rate.
    """

    def __init__(
            self,
            src_vocab_size: int,
            tgt_vocab_size: int,
            d_model: int = 256,
            n_heads: int = 8,
            n_layers: int = 3,
            max_len: int = 128,
            pad_idx: int = 0,
            dropout: float = 0.1,
            lr: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = Transformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_len=max_len,
            pad_idx=pad_idx,
            dropout=dropout,
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        self.lr = lr

    def forward(self, src, tgt):
        return self.model(src, tgt)

    def _compute_loss(self, batch):
        src = batch["src"]
        tgt = batch["tgt"]

        # Teacher forcing: input = [SOS] ... tokens, label = tokens ... [EOS]
        tgt_input = tgt[:, :-1]
        tgt_label = tgt[:, 1:]

        output = self.model(src, tgt_input)

        output = output.reshape(-1, output.size(-1))
        tgt_label = tgt_label.reshape(-1)

        return self.criterion(output, tgt_label)

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


def build_tokenizers(dataset, vocab_size: int = 8000):
    """Build source and target tokenizers from Multi30k dataset."""
    src_texts = [item["en"] for item in dataset]
    tgt_texts = [item["de"] for item in dataset]

    src_tokenizer = Tokenizer()
    tgt_tokenizer = Tokenizer()

    src_tokenizer.fit(src_texts, vocab_size=vocab_size)
    tgt_tokenizer.fit(tgt_texts, vocab_size=vocab_size)

    return src_tokenizer, tgt_tokenizer
