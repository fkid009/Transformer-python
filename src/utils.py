import re

class SimpleTokenizer:
    def __init__(self):
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.inv_vocab = {0: "<PAD>", 1: "<UNK>"}

    def normalize_and_split(self, text):
        """Normalizes and splits the input text into words."""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def build_vocab(self, texts):
        """Builds vocabulary from a list of texts."""
        idx = len(self.vocab)
        for sentence in texts:
            tokens = self.normalize_and_split(sentence)
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = idx
                    self.inv_vocab[idx] = token
                    idx += 1

    def encode(self, text):
        """Encodes the input text into a list of token IDs."""
        tokens = self.normalize_and_split(text)
        return [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
    
    def decode(self, token_ids):
        """Decodes a list of token IDs."""
        return [self.inv_vocab.get(token_id, "<UNK>") for token_id in token_ids]
    
    def pad(self, ids, max_len):
        return ids + [self.vocab["<PAD>"]] * (max_len - len(ids)) if len(ids) < max_len else ids[:max_len]
    
    def __call__(self, text, max_len=None):
        token_ids = self.encode(text)
        if max_len is not None:
            token_ids = self.pad(token_ids, max_len)
        return token_ids