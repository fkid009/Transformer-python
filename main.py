import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import load_dataset

from trainer import TranslationDataset, TransformerLitModule, build_tokenizers


def main():
    # 데이터 로드
    dataset = load_dataset("bentrevett/multi30k")
    train_data = dataset["train"]
    val_data = dataset["validation"]

    # 토크나이저 학습
    print("Building tokenizers...")
    src_tokenizer, tgt_tokenizer = build_tokenizers(train_data, vocab_size=8000)
    print(f"Source vocab size: {len(src_tokenizer.vocab)}")
    print(f"Target vocab size: {len(tgt_tokenizer.vocab)}")

    # 데이터셋 & 데이터로더
    max_len = 128
    batch_size = 64

    train_dataset = TranslationDataset(train_data, src_tokenizer, tgt_tokenizer, max_len)
    val_dataset = TranslationDataset(val_data, src_tokenizer, tgt_tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 모델
    model = TransformerLitModule(
        src_vocab_size=len(src_tokenizer.vocab),
        tgt_vocab_size=len(tgt_tokenizer.vocab),
        d_model=256,
        n_heads=8,
        n_layers=3,
        max_len=max_len,
        pad_idx=src_tokenizer.pad_id,
        dropout=0.1,
        lr=1e-4,
    )

    # 학습
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",
        precision="16-mixed",
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1),
            pl.callbacks.EarlyStopping(monitor="val_loss", patience=3, mode="min"),
        ],
        log_every_n_steps=50,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
