import torch
import wandb

import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F

from PIL.Image import Image

from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics.segmentation import MeanIoU
from torchmetrics.segmentation import GeneralizedDiceScore
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from config import Config
from models.lr_scheduler import CosineAnnealingWarmUpRestarts

class BasePLModels(pl.LightningModule):

    '''
    sample: dict[str, torch.Tensor] = {
        'img': torch.Tensor,  # Image tensor
        'ann': torch.Tensor,  # Annotation tensor
        'pred': torch.Tensor,  # Predicted output tensor
    }
    '''
    sample: dict[str, torch.Tensor] = {}
    # backbone: torch.nn.Module = None

    def __init__(self, backbone: torch.nn.Module, config: Config):
        super(BasePLModels, self).__init__()
        self.config = config
        self.backbone = backbone
        self.criterion: torch.nn.Module
        self.optimizer: torch.optim.Optimizer

    
    def __init_loggers(self):
        self.loggers: list[TensorBoardLogger, WandbLogger | None]
        self.tb_logger = self.loggers[0]

        if len(self.loggers) > 1:
            self.wandb_logger = self.loggers[1]
        else:
            self.wandb_logger = None

    
    def init_metrics(self):
        self.mIoU = MeanIoU(
            num_classes=self.config.model.num_classes,
        ).to(self.device)
        self.dice = GeneralizedDiceScore(
            num_classes=self.config.model.num_classes,
        ).to(self.device)
        self.preds_for_train_metrics = []
        self.targets_for_train_metrics = []
        self.preds_for_valid_metrics = []
        self.targets_for_valid_metrics = []

      
    def logits_to_pred(self, logits: torch.Tensor) -> torch.Tensor:
        if self.config.model.num_classes == 1:
            # Binary segmentation인 경우 sigmoid 적용
            prob = torch.sigmoid(logits)
            # BCEWithLogitsLoss의 입력은 float32이어야 함
            pred = (prob > self.config.model.threshold).float()
        else:
            # Multi-class segmentation인 경우 softmax 적용
            prob = F.softmax(logits, dim=1)
            # CrossEntropyLoss의 입력은 long이어야 함
            pred = torch.argmax(pred, dim=1).long()

        return pred
    

    def update_metrics(self, y_hat: torch.Tensor, y: torch.Tensor):
        '''
        Update metrics
        y_hat: torch.Tensor - Predicted output as indices
        (B, H, W)     # for binary segmentation
        or
        (B, N, H, W)
        y: torch.Tensor - Ground truth as indices
        (B, H, W)     # for binary segmentation
        or
        (B, N, H, W)
        '''
        y_hat = y_hat.detach().long()
        y = y.detach().long()
        # print(y_hat.shape, y.shape) # (B, C, H, W) torch.Size([4, 1, 64, 64]) torch.Size([4, 1, 64, 64])
        # y_hat: torch.Size([4, 1, 64, 64]) => y_hat: list[y_hat[0], y_hat[1], y_hat[2], ...]
        # y: torch.Size([4, 1, 64, 64]) => y: list[y[0], y[1], y[2], ...]
        if self.training:
            for i in range(y_hat.shape[0]):
                self.preds_for_train_metrics.append(y_hat[i])
                self.targets_for_train_metrics.append(y[i])
        else:
            for i in range(y_hat.shape[0]):
                self.preds_for_valid_metrics.append(y_hat[i])
                self.targets_for_valid_metrics.append(y[i])
            

    def calc_metrics(self, phase):
        if self.training:
            assert phase == 'train'
            preds_for_metrics = torch.cat(self.preds_for_train_metrics, dim=0)
            targets_for_metrics = torch.cat(self.targets_for_train_metrics, dim=0)
        else:
            assert phase == 'val'
            preds_for_metrics = torch.cat(self.preds_for_valid_metrics, dim=0)
            targets_for_metrics = torch.cat(self.targets_for_valid_metrics, dim=0)
        try:
            self.mIoU.update(preds_for_metrics, targets_for_metrics)
            self.dice.update(preds_for_metrics, targets_for_metrics)
            miou, dice = self.mIoU.compute(), self.dice.compute()
        except Exception as e:
            print(f'preds_for_metrics.shape: {preds_for_metrics.shape}, targets_for_metrics.shape: {targets_for_metrics.shape}')
            print(f'preds_for_metrics.unique: {preds_for_metrics.unique()}, targets_for_metrics.unique: {targets_for_metrics.unique()}')
            print(e)
            miou, dice = 0, 0
        finally:
            self.mIoU.reset()
            self.dice.reset()
            return miou, dice
    
    
    def log_metrics(self, phase: str):
        '''
        Log metrics
        '''
        miou, dice = self.calc_metrics(phase)
        metrics = {
            'mIoU': miou,
            'Dice': dice,
        }
        self._log_dict(
            metrics,
            prefix=phase,
        )


    def on_fit_start(self):
        if self.config.log:
            self.__init_loggers()
            # self.tb_logger.log_graph(self, torch.rand(1, 3, 512, 512))
            # self.wandb_logger.watch(self, log='all', log_freq=100)
        
        self.init_metrics()
        

    def _log_dict(self, kv: dict, prefix: str, prog_bar: bool=True):
        if not self.config.log:
            return
        for k, v in kv.items():
            self.log(f'{prefix}-{k}', v, prog_bar=prog_bar)

    
    def log_img(
            self,
            img: torch.Tensor | np.ndarray | Image,
            prefix: str,
            img_name: str
        ):
        if not self.config.log:
            return
        
        if isinstance(img, torch.Tensor):
            self._log_img_tensor(img, prefix, img_name)
        elif isinstance(img, np.ndarray):
            self._log_img_numpy(img, prefix, img_name)
        elif isinstance(img, Image):
            self._log_img_PIL(img, prefix, img_name)
        else:
            raise ValueError("img should be either torch.Tensor or PIL.Image")

    
    def _log_img_numpy(self, img: np.ndarray, prefix: str, img_name: str):
        tb_exp: SummaryWriter = self.tb_logger.experiment
        
        if len(img.shape) == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
            dataformats = 'HWC'
        elif len(img.shape) == 2:
            dataformats = 'HW'
        else:
            raise ValueError(f"img should have shape (H, W, C) or (H, W)\nimg.shape: {img.shape}")
        tb_exp.add_image(f'{prefix}/{img_name}', img, global_step=self.global_step, dataformats=dataformats)

        # Log image to wandb
        if self.wandb_logger:
            self.wandb_logger: WandbLogger
            self.wandb_logger.log_image(
                f'{prefix}/{img_name}',
                [wandb.Image(img, caption=img_name)],
                step=self.global_step
            )


    def _log_img_tensor(self, img: torch.Tensor, prefix: str, img_name: str):
        self._log_img_numpy(img.numpy(), prefix, img_name)

    
    def _log_img_PIL(self, img: Image, prefix: str, img_name: str):
        self._log_img_numpy(np.array(img), prefix, img_name)


    def save_samples(self, img: torch.Tensor, ann: torch.Tensor, pred: torch.Tensor):
        # Save sample images and annotations when sample dictionary is empty
        if not self.sample:
            self.sample['img'] = img
            self.sample['ann'] = ann
            self.sample['pred'] = pred
            return
        

    def on_train_epoch_start(self):
        self.preds_for_train_metrics.clear()
        self.targets_for_train_metrics.clear()
        self.mIoU.reset()
        self.dice.reset()


    def on_train_epoch_end(self) -> None:
        self.log_metrics(phase='train')


    def on_validation_epoch_start(self) -> None:
        self.preds_for_valid_metrics.clear()
        self.targets_for_valid_metrics.clear()
        self.mIoU.reset()
        self.dice.reset()
        
        
    
    def on_validation_epoch_end(self) -> None:       
        sample_img = self.sample['img']
        sample_ann = self.sample['ann']
        sample_pred = self.sample['pred']

        if len(sample_img.shape) == 2:
            sample_ann = sample_ann.unsqueeze(0)
            sample_pred = sample_pred.unsqueeze(0)

        assert len(sample_img.shape) == len(sample_ann.shape) == len(sample_pred.shape) == 3,\
            f"sample_img.shape: {sample_img.shape}, sample_ann.shape: {sample_ann.shape}, sample_pred.shape: {sample_pred.shape}"
        
        # sample_ann and sample_pred are in [0, self.config.model.num_classes - 1] range
        # Convert them to [0, 1] range
        if self.config.model.num_classes != 1:
            sample_ann = sample_ann / (self.config.model.num_classes - 1)
            sample_pred = sample_pred / (self.config.model.num_classes - 1)
        
        sample_img = (sample_img * 255).to(torch.uint8).detach().cpu()
        sample_ann = (sample_ann * 255).to(torch.uint8).detach().cpu()
        sample_pred = (sample_pred * 255).to(torch.uint8).detach().cpu()

        sample = torch.empty((3, sample_img.shape[1], sample_img.shape[2] * 3), dtype=torch.uint8, device='cpu')
        sample[:, :, :sample_img.shape[2]] = sample_img
        sample[:, :, sample_img.shape[2]:sample_img.shape[2] * 2] = sample_ann
        sample[:, :, sample_img.shape[2] * 2:] = sample_pred

        self.log_img(sample, 'val', 'sample')

        # Reset sample dictionary
        self.sample = {}

        self.log_metrics(phase='val')


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    

    def common_step(self, batch: torch.Tensor, batch_idx: int) -> dict[str, torch.Tensor]:
        raise NotImplementedError


    def configure_optimizers(self):
            optim = self.optimizer(self.parameters(), lr=self.config.training.lr)
            return optim
            # torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
            # lr_scheduler = CosineAnnealingWarmUpRestarts(
            #     optim,
            #     T_0=self.config.training.T_0,
            #     T_mult=self.config.training.T_mult,
            #     eta_max=self.config.training.eta_max,
            #     T_up=self.config.training.T_up,
            #     gamma=self.config.training.gamma,
            #     last_epoch=-1,
            # )
            # return [optim], [lr_scheduler]
    