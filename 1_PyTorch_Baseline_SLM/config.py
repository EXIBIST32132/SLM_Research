import os
from dataclasses import dataclass


@dataclass
class Config:
    # Data
    data_path: str = os.path.join("..", "0_Datasets", "medium_corpus.txt")
    block_size: int = 512  # context length (number of characters)
    batch_size: int = 48
    train_val_split: float = 0.9

    # Model
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.1

    # Training
    max_iters: int = 5000
    eval_interval: int = 500
    learning_rate: float = 3e-4
    device: str = "cuda"  # "cuda", "mps", or "cpu"
    num_workers: int = 2  # for DataLoader
    pin_memory: bool = True
    max_eval_batches: int = 8  # cap eval to keep it fast
    use_compile: bool = True  # torch.compile for speed (PyTorch 2.x)
    use_amp: bool = True  # mixed precision on CUDA/MPS

    # Generation
    sample_every: int = 500
    sample_tokens: int = 800
    temperature: float = 0.7
    top_k: int = 50

    # Checkpointing
    ckpt_dir: str = "checkpoints"
    save_interval: int = 400
    final_ckpt_name: str = "baseline_slm_final.pt"
