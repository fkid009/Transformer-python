# Transformer from Scratch

A minimal PyTorch implementation of the Transformer architecture from [Attention Is All You Need](https://arxiv.org/abs/1706.03762), trained on English-German translation (Multi30k).

## Project Structure

```
attention.py      Multi-Head Attention
embedding.py      Token Embedding + Positional Encoding
encoder.py        Encoder Layer & Encoder
decoder.py        Decoder Layer & Decoder
transformer.py    Full Transformer model
tokenizer.py      WordPiece Tokenizer
trainer.py        PyTorch Lightning module & dataset
main.py           Training entry point
```

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

## Hyperparameters

| Parameter | Default |
|-----------|---------|
| d_model | 256 |
| n_heads | 8 |
| n_layers | 3 |
| vocab_size | 4000 |
| max_len | 128 |
| batch_size | 64 |
| lr | 1e-4 |
| max_epochs | 10 |

Edit `main.py` to change hyperparameters.
