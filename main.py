import logging

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import load_dataset

from trainer import TranslationDataset, TransformerLitModule, build_tokenizers

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Hyperparameters ──
VOCAB_SIZE = 4000
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 3
MAX_LEN = 128
BATCH_SIZE = 64
DROPOUT = 0.1
LR = 1e-4
MAX_EPOCHS = 10


def main():
    # ── Step 1: Load dataset ──
    logger.info("Loading Multi30k dataset...")
    dataset = load_dataset("bentrevett/multi30k")
    train_data = dataset["train"]
    val_data = dataset["validation"]
    logger.info(f"Train: {len(train_data)} samples, Val: {len(val_data)} samples")

    # ── Step 2: Build tokenizers ──
    logger.info(f"Building tokenizers (vocab_size={VOCAB_SIZE})...")
    src_tokenizer, tgt_tokenizer = build_tokenizers(train_data, vocab_size=VOCAB_SIZE)
    logger.info(f"Source vocab size: {len(src_tokenizer.vocab)}")
    logger.info(f"Target vocab size: {len(tgt_tokenizer.vocab)}")

    # ── Step 3: Create dataloaders ──
    logger.info(f"Creating dataloaders (max_len={MAX_LEN}, batch_size={BATCH_SIZE})...")
    train_dataset = TranslationDataset(train_data, src_tokenizer, tgt_tokenizer, MAX_LEN)
    val_dataset = TranslationDataset(val_data, src_tokenizer, tgt_tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Step 4: Initialize model ──
    logger.info("Initializing Transformer model...")
    model = TransformerLitModule(
        src_vocab_size=len(src_tokenizer.vocab),
        tgt_vocab_size=len(tgt_tokenizer.vocab),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        max_len=MAX_LEN,
        pad_idx=src_tokenizer.pad_id,
        dropout=DROPOUT,
        lr=LR,
    )
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    # ── Step 5: Train ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info("Starting training...")
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        precision="16-mixed",
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1),
            pl.callbacks.EarlyStopping(monitor="val_loss", patience=3, mode="min"),
        ],
        log_every_n_steps=50,
    )

    trainer.fit(model, train_loader, val_loader)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
