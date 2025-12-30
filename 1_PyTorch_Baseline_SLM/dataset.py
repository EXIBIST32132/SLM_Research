import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from config import Config


class CharTokenizer:
    """
    Very simple character-level tokenizer.
    """

    def __init__(self, text: str):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s: str):
        return [self.stoi[c] for c in s]

    def decode(self, idxs):
        return "".join([self.itos[i] for i in idxs])


class TextDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int):
        super().__init__()
        self.data = data
        self.block_size = block_size

    def __len__(self):
        # Ensure non-negative length even if data is shorter than a full block
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return x, y


def load_text(path: str) -> str:
    if not os.path.isabs(path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.normpath(os.path.join(base_dir, path))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def create_dataloaders(config: Config) -> Tuple[CharTokenizer, DataLoader, DataLoader]:
    text = load_text(config.data_path)
    tokenizer = CharTokenizer(text)

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(len(data) * config.train_val_split)
    train_data = data[:n]
    val_data = data[n:]

    train_ds = TextDataset(train_data, config.block_size)
    val_ds = TextDataset(val_data, config.block_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=True,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
    )

    return tokenizer, train_loader, val_loader
