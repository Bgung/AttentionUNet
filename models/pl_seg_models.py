import torch

import numpy as np

from config import Config
from models.implemented.UNet import UNet, AttentionUNet, MHAUNet
from models.implemented.UNet_2Plus import UNet_2Plus
from models.implemented.UNet_3Plus import UNet_3Plus
from models.pl_base_model import BasePLModels

class PLSegModels(BasePLModels):

    def __init__(
            self,
            backbone: torch.nn.Module,
            criterion: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            config: Config
        ):
        super(PLSegModels, self).__init__(backbone, config)
        self.criterion: torch.nn.Module = criterion
        self.optimizer: torch.optim.Optimizer = optimizer


    def common_step(self, batch: torch.Tensor, batch_idx: int):
        img, ann = batch['img'], batch['ann']
        img: torch.Tensor; ann: torch.Tensor

        # if Model has an attention mechanism
        # logits, (embbed, (attention_map_1, ...))
        # else
        # logits, embbed
        logits, embbed = self(img)

        pred = self.logits_to_pred(logits)

        if getattr(self, 'attention_map_sample', None) is not None and\
                len(getattr(self, 'attention_map_sample')) == 0:
            self.attention_map_sample = embbed[1]

        if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss):
            ann = ann.float()

        loss = self.criterion(logits, ann)
        
        self.update_metrics(pred, ann)

        # save sample images and annotations
        self.save_samples(img[0], ann[0], pred[0])
        
        return {'loss': loss}
    
    
    def training_step(self, batch, batch_idx):
        loss_dict = self.common_step(batch, batch_idx)
        self._log_dict(
            loss_dict,
            'train',
        )
        return loss_dict['loss']


    def validation_step(self, batch, batch_idx):
        loss_dict = self.common_step(batch, batch_idx)
        self._log_dict(
            loss_dict,
            'val',
        )
        return loss_dict['loss']


class UNetPLSEG(PLSegModels):
    def __init__(
            self,
            in_channels: int=3,
            n_classes: int=1, # same as output_channels => Number of classes
            criterion: torch.nn.Module=torch.nn.BCELoss(),
            optimizer: torch.optim.Optimizer=torch.optim.Adam,
            config: Config=None
    ):
        super(UNetPLSEG, self).__init__(
            backbone=UNet(
                in_channels=in_channels,
                n_classes=n_classes
            ),
            criterion=criterion,
            optimizer=optimizer,
            config=config
        )
    
    

class UNet2PlusPLSEG(PLSegModels):
    def __init__(
            self,
            in_channels: int=3,
            n_classes: int=1, # same as output_channels
            criterion: torch.nn.Module=torch.nn.BCELoss(),
            optimizer: torch.optim.Optimizer=torch.optim.Adam,
            config: Config=None
    ):
        super(UNet2PlusPLSEG, self).__init__(
            backbone=UNet_2Plus(
                in_channels=in_channels,
                n_classes=n_classes
            ),
            criterion=criterion,
            optimizer=optimizer,
            config=config
        )

    
class UNet3PlusPLSEG(PLSegModels):
    def __init__(
            self,
            in_channels: int=3,
            n_classes: int=1, # same as output_channels
            criterion: torch.nn.Module=torch.nn.BCELoss(),
            optimizer: torch.optim.Optimizer=torch.optim.Adam,
            config: Config=None
    ):
        super(UNet3PlusPLSEG, self).__init__(
            backbone=UNet_3Plus(
                in_channels=in_channels,
                n_classes=n_classes
            ),
            criterion=criterion,
            optimizer=optimizer,
            config=config
        )


class AttentionUNetPLSEG(PLSegModels):
    def __init__(
            self,
            in_channels: int=3,
            n_classes: int=1, # same as output_channels
            criterion: torch.nn.Module=torch.nn.BCELoss(),
            optimizer: torch.optim.Optimizer=torch.optim.Adam,
            config: Config=None
    ):
        super(AttentionUNetPLSEG, self).__init__(
            backbone=AttentionUNet(
                in_channels=in_channels,
                n_classes=n_classes,
                attn_F_int=config.model.attn_F_int,
            ),
            criterion=criterion,
            optimizer=optimizer,
            config=config
        )
        self.attention_map_sample: tuple[torch.Tensor] = ()


    def visualize_attention_map(self, attention_map, name='attention_map'):
        """
        Visualize an attention map.
        Args:
            attention_map: Tensor of shape [C, H, W] or [H, W].
        """
        # If tensor has 3 dimensions, calculate mean across channels
        if len(attention_map.shape) == 3:
            attention_map = torch.mean(attention_map, dim=0)  # [H, W]
        
        # Convert to numpy array
        attention_map_np = attention_map.cpu().detach().numpy()

        # Normalize to [0, 1]
        attention_map_np = (attention_map_np - attention_map_np.min()) / (attention_map_np.max() - attention_map_np.min())
        self.log_img(
            (attention_map_np * 255).astype(np.uint8),
            'train' if self.training else 'val',
            name,
        )


    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        if self.config.log:
            # Log attention map from the last attention block
            # self.attention_map_sample is a tuple of (attention_map_1, ...)
            # and each attention_map_i is a tensor of shape [B, C, H, W]
            self.visualize_attention_map(self.attention_map_sample[-1][0], f'attention_map_{len(self.attention_map_sample)}')
            self.sample_attention_maps = () # Reset sample attention maps


class MHAUNetPLSEG(AttentionUNetPLSEG):
    def __init__(
            self,
            in_channels: int=3,
            n_classes: int=1, # same as output_channels
            criterion: torch.nn.Module=torch.nn.BCELoss(),
            optimizer: torch.optim.Optimizer=torch.optim.Adam,
            config: Config=None
    ):
        super(MHAUNetPLSEG, self).__init__(
            in_channels=in_channels,
            n_classes=n_classes,
            criterion=criterion,
            optimizer=optimizer,
            config=config
        )
        self.backbone=MHAUNet(
            in_channels=in_channels,
            n_classes=n_classes,
            attn_F_int=config.model.attn_F_int,
            attn_num_heads=config.model.attn_num_heads,
            attn_positional_encoding=config.model.attn_positional_encoding,
            input_size=config.dataset.input_shape
        )
        self.attention_map_sample: tuple[torch.Tensor] = ()