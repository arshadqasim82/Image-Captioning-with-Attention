import re
from collections import Counter
import json


class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.pad = "<pad>"
        self.start = "<start>"
        self.end = "<end>"
        self.unk = "<unk>"

        self.w2i = {
            self.pad: 0,
            self.start: 1,
            self.end: 2,
            self.unk: 3
        }
        self.i2w = {v: k for k, v in self.w2i.items()}
        self.idx = 4
        self.word_count = Counter()

    def clean_text(self, text):
        text = text.lower().strip()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.split()

    def build_vocab(self, captions):
        for caption in captions:
            words = self.clean_text(caption)
            self.word_count.update(words)
        for word, count in self.word_count.items():
            if count >= self.freq_threshold:
                self.w2i[word] = self.idx
                self.i2w[self.idx] = word
                self.idx += 1

    def encode(self, text):
        words = self.clean_text(text)
        encoded = [self.w2i[self.start]]
        for word in words:
            encoded.append(self.w2i.get(word, self.w2i[self.unk]))
        encoded.append(self.w2i[self.end])
        return encoded

    def decode(self, token_ids):
        words = []
        for token in token_ids:
            word = self.i2w.get(token, self.unk)
            if word == self.end:
                break
            if word not in [self.pad, self.start]:
                words.append(word)

        return " ".join(words)

    def save(self, filepath):
        data = {
            "freq_threshold": self.freq_threshold,
            "idx": self.idx,
            "w2i": self.w2i,
            "i2w": self.i2w,
            "word_counts": dict(self.word_counts)
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Vocabulary saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'Vocabulary':
        with open(filepath, 'r') as f:
            data = json.load(f)

        vocab = cls(freq_threshold=data["freq_threshold"])
        vocab.idx = data["idx"]
        vocab.w2i = data["w2i"]

        vocab.i2w = {int(k): v for k, v in data["i2w"].items()}
        vocab.word_counts = Counter(data["word_counts"])

        return vocab
