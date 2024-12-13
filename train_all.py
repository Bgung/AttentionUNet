import torch
import random

import numpy as np
import pytorch_lightning as pl

from config import Config
from datasets.BaseDataset import BaseDataset
from models.pl_base_model import BasePLModels
from models.pl_seg_models import (
    UNetPLSEG,
    UNet2PlusPLSEG,
    UNet3PlusPLSEG,
    AttentionUNetPLSEG,
    MHAUNetPLSEG
)
from models.pl_ctl_models import (
    UNetPLCTL,
    UNet2PlusPLCTL,
    UNet3PlusPLCTL,
    AttentionUNetPLCTL,
    MHAUNetPLCTL
)
from datasets import (
    CHASEDB1,
    EBHI,
    HRF,
    KVASIR
)
from train_seg import main as train_seg
from train_contrastive import main as train_contrastive


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed, workers=True)
    np.random.seed(seed)
    random.seed(seed)


def main():
    models_seg = [
        MHAUNetPLSEG,
        AttentionUNetPLSEG,
        UNetPLSEG,
        # UNet2PlusPLSEG,
        # UNet3PlusPLSEG
    ]
    models_ctl = [
        MHAUNetPLCTL,
        AttentionUNetPLCTL,
        UNetPLCTL,
        # UNet2PlusPLCTL,
        # UNet3PlusPLCTL
    ]
    datasets = [
        # CHASEDB1,
        EBHI,
        HRF,
        KVASIR
    ]


    pairs_seg = [
        (dataset, model)
        for dataset in datasets
        for model in models_seg
    ]

    temperatures = [
        0.01,
        # 0.1,
        # 1.0,
    ]

    pairs_contrastive = [
        (dataset, model, temperature)
        for dataset in datasets
        for model in models_ctl
        for temperature in temperatures
    ]


    config = Config()
    config.tags = ['v2']

    seed_everything(config.seed)

    train_seg(pairs_seg, config)
    train_contrastive(pairs_contrastive, config)

if __name__=='__main__':
    main()