import os
import random

from tokenizer import CharTokenizer


class TextData:
    def __init__(self, input_path, block_size, split_ratio=0.9):
        if not os.path.exists(input_path):
            print("Error: dataset file not found at %s" % input_path)
            raise SystemExit(1)
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        if not text:
            print("Error: dataset file is empty.")
            raise SystemExit(1)

        self.tokenizer = CharTokenizer()
        self.tokenizer.build_vocab(text)
        ids = self.tokenizer.encode(text)
        if len(ids) <= block_size:
            print("Error: dataset file is too small for the configured block_size.")
            raise SystemExit(1)

        split = int(len(ids) * split_ratio)
        if split <= 0 or split >= len(ids):
            split = len(ids) - block_size - 1
        self.train_ids = ids[:split]
        self.val_ids = ids[split:]
        self.block_size = block_size

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def get_batch(self, split, batch_size):
        data = self.train_ids if split == "train" else self.val_ids
        max_start = len(data) - self.block_size - 1
        if max_start <= 0:
            raise ValueError("Not enough data to create a batch.")
        x_batch = []
        y_batch = []
        for _ in range(batch_size):
            idx = random.randint(0, max_start)
            chunk = data[idx : idx + self.block_size + 1]
            x_batch.append(chunk[:-1])
            y_batch.append(chunk[1:])
        return x_batch, y_batch
