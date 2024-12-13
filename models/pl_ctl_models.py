import torch

import numpy as np
import torch.nn.functional as F

from config import Config
from models.loss import ContrastiveLoss, PixelContrastLoss
from models.implemented import UNet, UNet_2Plus, UNet_3Plus, AttentionUNet, MHAUNet
from models.pl_base_model import BasePLModels

class ProjectHead(torch.nn.Module):
    def __init__(self, in_dim: int, embedding_dim: int):
        super(ProjectHead, self).__init__()
        self.projection = torch.nn.Sequential(
            torch.nn.Conv2d(in_dim, in_dim, kernel_size=1),
            torch.nn.BatchNorm2d(in_dim),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_dim, embedding_dim, kernel_size=1),
        )
    
    def forward(self, x):
        x = self.projection(x)
        return F.normalize(x, p=2, dim=1)
    
def calculate_convtranspose_steps(input_size, target_size, kernel_size, stride, padding, output_padding):
    """
    Calculate the number of ConvTranspose2D operations required to transform
    an input image size to a target image size.
    
    Args:
    - input_size (tuple): (H_in, W_in), the input image size.
    - target_size (tuple): (H_target, W_target), the desired output image size.
    - kernel_size (int): The size of the convolutional kernel.
    - stride (int): The stride of the ConvTranspose2D operation.
    - padding (int): The padding applied to the input.
    - output_padding (int): The additional size added to the output size.
    
    Returns:
    - steps (int): The number of ConvTranspose2D layers required.
    """
    H_in, W_in = input_size
    H_target, W_target = target_size
    
    steps = 0
    while (H_in < H_target or W_in < W_target):
        H_in = (H_in - 1) * stride - 2 * padding + kernel_size + output_padding
        W_in = (W_in - 1) * stride - 2 * padding + kernel_size + output_padding
        steps += 1
        
        # Break if overshooting target size (error in settings)
        if H_in > H_target or W_in > W_target:
            raise ValueError("Overshooting target size! Check your kernel_size, stride, padding, or output_padding.")
    
    return steps


def calculate_output_shape(input_shape, num_convs, kernel_size=3, stride=1, padding=1):
    """
    Calculate the output shape of a series of convolutional layers.
    
    input_shape: tuple, (채널 수, 높이, 너비)
    num_convs: int, 컨볼루션 레이어 수
    kernel_size: int, 기본값 3
    stride: int, 기본값 1
    padding: int, 기본값 1

    Returns: tuple, 출력 형태
    """
    channels, height, width = input_shape
    for _ in range(num_convs):
        height = (height + 2 * padding - kernel_size) // stride + 1
        width = (width + 2 * padding - kernel_size) // stride + 1
    return (channels, height, width)


class PLCTLModels(BasePLModels):
    
    def __init__(
            self,
            backbone: torch.nn.Module,
            seg_criterion: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            config: Config
        ):
        super(PLCTLModels, self).__init__(backbone, config)
        self.seg_criterion: torch.nn.Module = seg_criterion
        self.ctl_criterion: torch.nn.Module = PixelContrastLoss(config.model.temperature)
        self.optimizer: torch.optim.Optimizer = optimizer

        self.criterion = ContrastiveLoss(seg_criterion, self.ctl_criterion)

        self.proj_head = ProjectHead(
            in_dim=self.backbone.filters[0],
            embedding_dim=config.model.proj_dim
        )
        self.upsample = torch.nn.Upsample(
            size=self.config.dataset.input_shape,
            mode='bilinear',
            align_corners=True
        )


    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        return super().forward(x)
    

    def common_step(self, batch: torch.Tensor, batch_idx: int):
        img, ann = batch['img'], batch['ann']
        logits, embedding = self(img)

        if isinstance(embedding, tuple):
            embedding, attention_maps = embedding
        
        embedding = self.proj_head(embedding)
        embedding = self.upsample(embedding)

        pred = self.logits_to_pred(logits)

        if getattr(self, 'attention_map_sample', None) is not None and\
                len(getattr(self, 'attention_map_sample')) == 0:
            self.attention_map_sample = attention_maps

        if isinstance(self.criterion, torch.nn.BCEWithLogitsLoss):
            ann = ann.float()

        loss_dict = self.criterion(
            embed=embedding,
            pred=logits,
            target=ann
        )
        
        self.update_metrics(pred, ann)

        # save sample images and annotations
        self.save_samples(img[0], ann[0], pred[0])
        
        return loss_dict
    
    
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


class UNetPLCTL(PLCTLModels):
    def __init__(
            self,
            in_channels: int=3,
            n_classes: int=1,
            seg_criterion: torch.nn.Module=torch.nn.BCELoss(),
            optimizer: torch.optim.Optimizer=torch.optim.Adam,
            config: Config=None
    ):
        super(UNetPLCTL, self).__init__(
            UNet(
                in_channels=in_channels,
                n_classes=n_classes,
            ),
            seg_criterion,
            optimizer,
            config
        )
        self.optimizer = optimizer


class UNet2PlusPLCTL(PLCTLModels):
    def __init__(
            self,
            in_channels: int=3,
            n_classes: int=1, # same as output_channels
            seg_criterion: torch.nn.Module=torch.nn.BCELoss(),
            optimizer: torch.optim.Optimizer=torch.optim.Adam,
            config: Config=None
    ):
        super(UNet2PlusPLCTL, self).__init__(
            UNet_2Plus(
                in_channels=in_channels,
                n_classes=n_classes,
            ),
            seg_criterion,
            optimizer,
            config
        )
        self.optimizer = optimizer


class UNet3PlusPLCTL(PLCTLModels):
    def __init__(
            self,
            in_channels: int=3,
            n_classes: int=1, # same as output_channels
            seg_criterion: torch.nn.Module=torch.nn.BCELoss(),
            optimizer: torch.optim.Optimizer=torch.optim.Adam,
            config: Config=None
    ):
        super(UNet3PlusPLCTL, self).__init__(
            UNet_3Plus(
                in_channels=in_channels,
                n_classes=n_classes,
            ),
            seg_criterion,
            optimizer,
            config
        )
        self.optimizer = optimizer


class AttentionUNetPLCTL(PLCTLModels):
    def __init__(
            self,
            in_channels: int=3,
            n_classes: int=1, # same as output_channels
            seg_criterion: torch.nn.Module=torch.nn.BCELoss(),
            optimizer: torch.optim.Optimizer=torch.optim.Adam,
            config: Config=None
    ):
        super(AttentionUNetPLCTL, self).__init__(
            backbone=AttentionUNet(
                in_channels=in_channels,
                n_classes=n_classes,
                attn_F_int=config.model.attn_F_int,
            ),
            seg_criterion=seg_criterion,
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
            self.sample_attention_maps = () # Reset attention map sample


class MHAUNetPLCTL(AttentionUNetPLCTL):
    def __init__(
            self,
            in_channels: int=3,
            n_classes: int=1, # same as output_channels
            seg_criterion: torch.nn.Module=torch.nn.BCELoss(),
            optimizer: torch.optim.Optimizer=torch.optim.Adam,
            config: Config=None
    ):
        super(MHAUNetPLCTL, self).__init__(
            in_channels=in_channels,
            n_classes=n_classes,
            seg_criterion=seg_criterion,
            optimizer=optimizer,
            config=config
        )
        self.backbone=MHAUNet(
            in_channels=in_channels,
            n_classes=n_classes,
            attn_F_int=config.model.attn_F_int,
            attn_num_heads=config.model.attn_num_heads,
            attn_positional_encoding=config.model.attn_positional_encoding,
            input_size=config.dataset.input_shape,
        )
        self.attention_map_sample: tuple[torch.Tensor] = ()
