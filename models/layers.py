import torch

from torch import nn
from torch.functional import F

from torchvision.transforms.functional import center_crop

def get_final_activation(final_activation: str) -> nn.Module:
    if final_activation == "sigmoid":
        return nn.Sigmoid()
    elif final_activation == "softmax":
        return nn.Softmax(dim=1)
    elif final_activation == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Invalid final activation: {final_activation}")
    
class DoubleConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int=3,
            stride: int=1,
            padding: int=1
        ):
        super(DoubleConv, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class DownSample(nn.Module):
    def __init__(self):
        super(DownSample, self).__init__()

        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_pool(x)
    
class UpSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(UpSample, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)
    
class CropAndConcat(nn.Module):
    def __init__(self):
        super(CropAndConcat, self).__init__()

    def forward(self, x_expanding: torch.Tensor, x_contracting: torch.Tensor) -> torch.Tensor:
        x_contracting = center_crop(
            x_contracting, (x_expanding.shape[2], x_expanding.shape[3])
        )

        return torch.cat(
            [x_expanding, x_contracting], dim=1
        )