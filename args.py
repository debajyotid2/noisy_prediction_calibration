"""
Argument interfacing with Hydra configurations
"""

from dataclasses import dataclass

@dataclass
class TrainingArgs:
    num_epochs: int
    learning_rate: float
    seed: int
    log_dir: str
    latent_dim: int


@dataclass
class DatasetArgs:
    img_height: int
    img_width: int
    n_channels: int
    batch_size: int
    data_size: int
    num_classes: int
    val_frac: float
    name: str


@dataclass
class Args:
    training: TrainingArgs
    dataset: DatasetArgs
