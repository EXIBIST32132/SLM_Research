from dataclasses import dataclass, asdict


@dataclass
class Config:
    block_size: int = 32
    batch_size: int = 8
    embed_dim: int = 16
    hidden_dim: int = 32
    n_layers: int = 1
    learning_rate: float = 0.003
    max_steps: int = 200
    eval_interval: int = 50
    sample_every: int = 100
    sample_tokens: int = 120
    seed: int = 1337
    checkpoint_path: str = "checkpoints/model.pkl"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)
