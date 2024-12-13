import albumentations as A

from dataclasses import dataclass, field

@dataclass
class Dataset:
    root: str="./data"
    dataset_name: str="CHASEDB1"

    input_shape: tuple=(64, 64)
    transforms: list=field(default_factory=lambda: [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.1, 0.1), shear=10, p=0.5),
        A.Normalize(mean=(0.5,), std=(0.5,)),  # Normalize image
    ])

    
@dataclass
class Model:
    model_name: str="UNet"
    in_channels: int=3
    num_classes: int=1
    threshold: float=0.5

    proj_dim: int=32

    temperature: float=0.01 # Contrastive loss temperature

    attn_F_int: int=16
    attn_num_heads: int=4
    attn_positional_encoding: bool=True

@dataclass
class Training:
    batch_size: int=2
    epochs: int=100
    num_workers: int=4
    valid_ratio: float=0

    lr: float=1e-3
    
    # # LR Scheduler
    # T_0: int=10
    # T_mult: int=1
    # eta_max: float=0.1
    # T_up: int=10
    # gamma: float=0.5

    # Early Stopping
    # patience: int=30
    # monitor: str="val-loss"
    # mode: str="min"
    # min_delta: float=0.001
    # verbose: bool=True


@dataclass
class Config:
    log: bool=True
    seed: int=42
    tags: list | None=None

    dataset: Dataset = field(default_factory=Dataset)
    model: Model = field(default_factory=Model)
    training: Training = field(default_factory=Training)
