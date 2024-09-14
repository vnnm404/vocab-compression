from dataclasses import dataclass, field
from typing import Dict


@dataclass
class DataConfig:
    name: str = "tinystories"
    num_workers: int = 4
    batch_size: int = 32
    max_length: int = 512


@dataclass
class TokenizerConfig:
    name: str = "gpt2"


@dataclass
class ModelConfig:
    name: str = "simple-proto"
    gpt2: Dict[str, int] = field(
        default_factory=lambda: {"hidden_size": 128, "layers": 8, "heads": 8}
    )


@dataclass
class TrainingConfig:
    epochs: int = 1
    lr: float = 1e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.01


@dataclass
class Config:
    cache_dir: str = ".cache"
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


config = Config()
