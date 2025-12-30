import json


class CharTokenizer:
    def __init__(self):
        self.stoi = {}
        self.itos = []

    def build_vocab(self, text):
        chars = sorted(set(text))
        self.itos = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    def encode(self, text):
        ids = []
        for ch in text:
            if ch not in self.stoi:
                raise ValueError("Character not in vocab: %r" % ch)
            ids.append(self.stoi[ch])
        return ids

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def save_vocab(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"itos": self.itos}, f)

    def load_vocab(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.itos = list(data["itos"])
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}
        return self

    @property
    def vocab_size(self):
        return len(self.itos)
