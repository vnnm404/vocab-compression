from dataclasses import dataclass, field
from typing import Dict
import sys

@dataclass
class DataConfig:
    name: str = "tinystories"
    num_workers: int = 32
    batch_size: int = 32
    max_length: int = 512


@dataclass
class TokenizerConfig:
    name: str = "gpt2"


@dataclass
class ModelConfig:
    name: str = "simple-eos-optim-test"
    gpt2: Dict[str, int] = field(
        default_factory=lambda: {"hidden_size": 64, "layers": 4, "heads": 2}
    )
    gptneo: Dict[str, int] = field(default_factory=lambda: {"window_size": 256})
    compression: Dict[str, int] = field(default_factory=lambda: {"group_size": 64})


@dataclass
class TrainingConfig:
    epochs: int = 1
    lr: float = 1e-4
    run_name: str = 'vocab_comp'
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
