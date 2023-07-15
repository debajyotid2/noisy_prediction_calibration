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
    model_dim: int
    dropout_p: float


@dataclass
class DatasetArgs:
    img_height: int
    img_width: int
    n_channels: int
    batch_size: int
    data_size: int
    n_classes: int
    val_frac: float
    name: str

@dataclass
class NPCArgs:
    softplus_beta: float
    n_neighbors: int
    num_epochs: int
    kld_reg: float
    prior_norm: float
    clipnorm: float


@dataclass
class Args:
    training: TrainingArgs
    dataset: DatasetArgs
    npc: NPCArgs
