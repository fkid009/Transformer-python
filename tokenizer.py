import re
from collections import Counter


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[SOS]", "[EOS]"]
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")


class Tokenizer:
    def __init__(self, vocab=None, lowercase=True):
        self.lowercase = lowercase
        self.vocab = self._prepare_vocab(vocab)
        self._build_vocab_map()

    def _prepare_vocab(self, vocab):
        if vocab is None:
            return SPECIAL_TOKENS.copy()

        merged = SPECIAL_TOKENS.copy()
        for token in vocab:
            if token not in merged:
                merged.append(token)
        return merged

    def _build_vocab_map(self):
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}
        self.pad_id = self.token_to_id["[PAD]"]
        self.unk_id = self.token_to_id["[UNK]"]
        self.sos_id = self.token_to_id["[SOS]"]
        self.eos_id = self.token_to_id["[EOS]"]

    def _normalize(self, text):
        return text.lower() if self.lowercase else text

    def _pre_tokenize(self, text):
        return TOKEN_PATTERN.findall(self._normalize(text))

    def _split_word(self, word):
        if not word:
            return []

        pieces = [word[0]]
        pieces.extend("##" + char for char in word[1:])
        return pieces

    def _build_word_freqs(self, texts):
        word_freqs = Counter()

        for text in texts:
            for word in self._pre_tokenize(text):
                word_freqs[word] += 1

        return word_freqs

    def _compute_pair_scores(self, splits, word_freqs):
        pair_freqs = Counter()
        letter_freqs = Counter()

        for word, freq in word_freqs.items():
            pieces = splits[word]

            if len(pieces) == 1:
                letter_freqs[pieces[0]] += freq
                continue

            for i in range(len(pieces) - 1):
                pair = (pieces[i], pieces[i + 1])
                pair_freqs[pair] += freq
                letter_freqs[pieces[i]] += freq

            letter_freqs[pieces[-1]] += freq

        scores = {}
        for pair, pair_freq in pair_freqs.items():
            left, right = pair
            scores[pair] = pair_freq / (letter_freqs[left] * letter_freqs[right])

        return scores

    def _merge_pair(self, left, right, splits, word_freqs):
        merged_token = left + right.replace("##", "")

        for word in word_freqs:
            pieces = splits[word]
            if len(pieces) < 2:
                continue

            new_pieces = []
            i = 0

            while i < len(pieces):
                if i < len(pieces) - 1 and pieces[i] == left and pieces[i + 1] == right:
                    new_pieces.append(merged_token)
                    i += 2
                else:
                    new_pieces.append(pieces[i])
                    i += 1

            splits[word] = new_pieces

        return merged_token

    def fit(self, texts, vocab_size=100):
        word_freqs = self._build_word_freqs(texts)
        splits = {word: self._split_word(word) for word in word_freqs}

        vocab = set(SPECIAL_TOKENS)
        for pieces in splits.values():
            vocab.update(pieces)

        while len(vocab) < vocab_size:
            scores = self._compute_pair_scores(splits, word_freqs)
            if not scores:
                break

            best_pair = max(scores, key=scores.get)
            merged_token = self._merge_pair(best_pair[0], best_pair[1], splits, word_freqs)

            if merged_token in vocab:
                continue

            vocab.add(merged_token)

        self.vocab = SPECIAL_TOKENS + sorted(vocab - set(SPECIAL_TOKENS))
        self._build_vocab_map()

    def tokenize_word(self, word):
        pieces = []
        start = 0

        while start < len(word):
            end = len(word)
            match = None

            while end > start:
                token = word[start:end]
                if start > 0:
                    token = "##" + token

                if token in self.token_to_id:
                    match = token
                    break

                end -= 1

            if match is None:
                return ["[UNK]"]

            pieces.append(match)
            start = end

        return pieces

    def tokenize(self, text):
        tokens = []

        for word in self._pre_tokenize(text):
            tokens.extend(self.tokenize_word(word))

        return tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.token_to_id.get(token, self.unk_id) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.id_to_token.get(idx, "[UNK]") for idx in ids]

    def encode(self, text, add_special_tokens=True):
        tokens = self.tokenize(text)

        if add_special_tokens:
            tokens = ["[SOS]"] + tokens + ["[EOS]"]

        return self.convert_tokens_to_ids(tokens)

    def decode(self, ids, skip_special_tokens=True):
        tokens = self.convert_ids_to_tokens(ids)
        words = []

        for token in tokens:
            if skip_special_tokens and token in SPECIAL_TOKENS:
                continue

            if token.startswith("##") and words:
                words[-1] += token[2:]
            else:
                words.append(token)

        text = " ".join(words)
        return re.sub(r"\s+([.,!?;:'\"])", r"\1", text)

    def pad(self, ids, max_length):
        ids = ids[:max_length]
        return ids + [self.pad_id] * (max_length - len(ids))
