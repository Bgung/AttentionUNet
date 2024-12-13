import torch
import wandb
import random

import numpy as np
import pytorch_lightning as pl

from config import Config
from datasets.BaseDataset import BaseDataset
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, Timer, LearningRateMonitor


def worker_seed_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def wandb_init(train):
    def init(*args, **kwargs):
        if kwargs["config"].log:
            wandb.init(
                project="MedicalAI2024",
                name=kwargs["config"].model.model_name,
                config=kwargs["config"],
                tags=kwargs["config"].tags
            )
        train(*args, **kwargs)
        if kwargs["config"].log:
            wandb.finish()
    return init


@wandb_init
def trainer(
    model: pl.LightningModule,
    dataset: BaseDataset,
    config: Config
):
    if config.training.valid_ratio > 0:
        train_size = int((1 - config.training.valid_ratio) * len(dataset))
        valid_size = len(dataset) - train_size

        train_dataset, valid_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, valid_size]
        )
    else:
        train_dataset = dataset
        valid_dataset = None

    setattr(train_dataset, 'is_train', True)
    if valid_dataset:
        setattr(valid_dataset, 'is_train', False)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_seed_fn
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_seed_fn
    ) if valid_dataset else None

    loggers = [
        TensorBoardLogger(
            "tb_logs",
            name=config.model.model_name,
            default_hp_metric=True,
            sub_dir=dataset.dataset_name
        ),
        WandbLogger(project="MedicalAI2024", name=config.model.model_name),
    ] if config.log else []

    callbacks = [
        # EarlyStopping(
        #     monitor=config.training.monitor,
        #     patience=config.training.patience,
        #     mode=config.training.mode,
        #     min_delta=config.training.min_delta,
        #     verbose=config.training.verbose
        # ),     
    ]
    if config.log:
        callbacks.append(LearningRateMonitor(logging_interval='step'))
        callbacks.append(Timer())

    trainer = pl.Trainer(
        max_epochs=config.training.epochs,
        log_every_n_steps=1,
        logger=loggers,
        callbacks=callbacks,
    )
    trainer.fit(model, train_loader, val_dataloaders=valid_loader)
