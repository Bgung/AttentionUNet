import math
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

from models.layers import get_final_activation
from models.implemented.layers import unetConv2, unetUp
from models.implemented.init_weights import init_weights

class UNet(nn.Module):

    def __init__(self, in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.n_classes = n_classes
        #
        # filters = [32, 64, 128, 256, 512]
        self.filters = filters = [64, 128, 256, 512, 1024]
        # # filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
        #
        self.outconv1 = nn.Conv2d(filters[0], self.n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def dotProduct(self,seg,cls):
        B, N, H, W = seg.size()
        seg = seg.view(B, N, H * W)
        final = torch.einsum("ijk,ij->ijk", [seg, cls])
        final = final.view(B, N, H, W)
        return final

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Args:
            inputs (torch.Tensor): input image tensor
        Returns:
            tuple[torch.Tensor, torch.Tensor]: segmentation map, embedding
        '''
        conv1 = self.conv1(inputs)  # 16*512*1024
        maxpool1 = self.maxpool1(conv1)  # 16*256*512

        conv2 = self.conv2(maxpool1)  # 32*256*512
        maxpool2 = self.maxpool2(conv2)  # 32*128*256

        conv3 = self.conv3(maxpool2)  # 64*128*256
        maxpool3 = self.maxpool3(conv3)  # 64*64*128

        conv4 = self.conv4(maxpool3)  # 128*64*128
        maxpool4 = self.maxpool4(conv4)  # 128*32*64

        center = self.center(maxpool4)  # 256*32*64
        
        # print("center", center.shape)           # 1024*32*64
        # print("conv4", conv4.shape)             # 512*64*128
        up4 = self.up_concat4(center, conv4)  # 128*64*128
        up3 = self.up_concat3(up4, conv3)  # 64*128*256
        up2 = self.up_concat2(up3, conv2)  # 32*256*512
        up1 = self.up_concat1(up2, conv1)  # 16*512*1024

        d1 = self.outconv1(up1)  # 256

        return d1, up1


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        기본 Attention Block 구현
        Args:
            F_g: Gate 신호의 입력 채널 수 (from Decoder)
            F_l: Skip Connection의 입력 채널 수 (from Encoder)
            F_int: 내부 채널 수 (중간 레이어에서 사용)
        """
        super(AttentionBlock, self).__init__()
        
        # Gate 신호를 변환하는 1x1 Conv
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Skip Connection을 변환하는 1x1 Conv
        # x^l shape -> [N, F_int, H_x, W_x] 에서 [N, F_int, H_g, W_g] 로 다운샘플링(서브샘플링)
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # Attention Map 계산을 위한 psi
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
        )
        
        # 활성화 함수
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, g, x):
        """
        Forward 함수
        Args:
            g: Gate 신호 (from Decoder) - [N, F_g, H, W]
            x: Skip Connection (from Encoder) - [N, F_l, H, W]
        Returns:
            중요도가 적용된 Skip Connection - [N, F_l, H, W]
        """
        # Gate 신호 변환
        g1 = self.W_g(g)  # [N, F_int, H_g, W_g]
        
        # Skip Connection 변환
        # x => [N, F_int, H_x, W_x]
        x1 = self.W_x(x)  # 
        # x1 => [N, F_int, H_g, W_g]

        # Gate 신호와 Skip Connection의 합산 후 ReLU 활성화
        attention_features = self.relu(g1 + x1)  # [N, F_int, H, W]
        
        # Attention Map 계산
        alpha = self.psi(attention_features)  # [N, 1, H, W]

        alpha = self.sigmoid(alpha)

        alpha = F.interpolate(alpha, size=x.size()[2:], mode='bilinear', align_corners=True)  # [N, 1, H_x, W_x]
        
        # Attention Map과 Skip Connection의 요소별 곱
        out = alpha * x  # [N, F_l, H, W]
        
        return out


class AttentionUNet(nn.Module):

    def __init__(
            self,
            in_channels=3,
            n_classes=1,
            is_deconv=True,
            is_batchnorm=True,
            attn_F_int=16,
        ):
        super(AttentionUNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.n_classes = n_classes
        #
        # filters = [32, 64, 128, 256, 512]
        self.filters = filters = [64, 128, 256, 512, 1024]
        # # filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # attention layers
        self.attn_1 = AttentionBlock(F_g=filters[4], F_l=filters[3], F_int=attn_F_int)
        self.attn_2 = AttentionBlock(F_g=filters[3], F_l=filters[2], F_int=attn_F_int)
        self.attn_3 = AttentionBlock(F_g=filters[2], F_l=filters[1], F_int=attn_F_int)
        self.attn_4 = AttentionBlock(F_g=filters[1], F_l=filters[0], F_int=attn_F_int)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        #
        self.outconv1 = nn.Conv2d(filters[0], self.n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def dotProduct(self,seg,cls):
        B, N, H, W = seg.size()
        seg = seg.view(B, N, H * W)
        final = torch.einsum("ijk,ij->ijk", [seg, cls])
        final = final.view(B, N, H, W)
        return final

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, tuple[torch.Tensor, ...]]]:
        '''
        Args:
            inputs (torch.Tensor): input image tensor
        Returns:
            tuple[torch.Tensor, tuple[torch.Tensor, tuple[torch.Tensor, ...]]]: segmentation map, (embedding, (attention maps for each layer))
        '''
        # print("inputs", inputs.shape)           # 3*512*512
        conv1 = self.conv1(inputs)
        # print("conv1", conv1.shape)             # 64*512*512
        maxpool1 = self.maxpool1(conv1)
        # print("maxpool1", maxpool1.shape)       # 64*256*256


        conv2 = self.conv2(maxpool1)
        # print("conv2", conv2.shape)             # 128*256*256
        maxpool2 = self.maxpool2(conv2)
        # print("maxpool2", maxpool2.shape)       # 128*128*128


        conv3 = self.conv3(maxpool2)
        # print("conv3", conv3.shape)             # 256*128*128
        maxpool3 = self.maxpool3(conv3)
        # print("maxpool3", maxpool3.shape)       # 256*64*64


        conv4 = self.conv4(maxpool3)
        # print("conv4", conv4.shape)             # 512*64*64
        maxpool4 = self.maxpool4(conv4)
        # print("maxpool4", maxpool4.shape)       # 512*32*32

        center = self.center(maxpool4)
        # print("center", center.shape)           # 1024*32*32

        # conv4 = (B, 512, 64, 64)
        attn_1 = self.attn_1(center, conv4)
        # print("attn_1", attn_1.shape)                         # 512*64*64
        up4 = self.up_concat4(center, attn_1)
        # print("up4", up4.shape)                               # 512*64*64

        attn_2 = self.attn_2(up4, conv3)
        # print("attn_2", attn_2.shape)                         # 256*128*128
        up3 = self.up_concat3(up4, attn_2)
        # print("up3", up3.shape)                               # 256*128*128

        attn_3 = self.attn_3(up3, conv2)
        # print("attn_3", attn_3.shape)                         # 128*256*256
        up2 = self.up_concat2(up3, attn_3)
        # print("up2", up2.shape)                               # 128*256*256

        attn_4 = self.attn_4(up2, conv1)
        # print("attn_4", attn_4.shape)                         # 64*512*512
        up1 = self.up_concat1(up2, attn_4)
        # print("up1", up1.shape)                               # 64*512*512

        d1 = self.outconv1(up1)
        # print("d1", d1.shape)                               # 1*512*512

        return d1, (up1, (attn_1, attn_2, attn_3, attn_4))


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        N, L, _ = x.size()
        return x + self.encoding[:, :L, :].to(x.device)



class MHABlock(nn.Module):
    def __init__(self, F_g, F_l, F_int, num_heads=4, positional_encoding: bool=True, max_len: int=512):
        """
        Multi Head Attention Block 구현
        Args:
            F_g: Gate 신호의 입력 채널 수 (from Decoder)
            F_l: Skip Connection의 입력 채널 수 (from Encoder)
            F_int: 내부 채널 수 (중간 레이어에서 사용)
        """
        super(MHABlock, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=F_int, num_heads=num_heads, batch_first=True
        )
        if positional_encoding:
            self.positional_encoding = PositionalEncoding(embed_dim=F_int, max_len=max_len)

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # Attention Map 계산을 위한 psi
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
        )

        self.sigmoid = nn.Sigmoid()

    
    def forward(self, g, x):
        """
        Forward 함수
        Args:
            g: Gate 신호 (from Decoder) - [N, F_g, H, W]
            x: Skip Connection (from Encoder) - [N, F_l, H, W]
        Returns:
            중요도가 적용된 Skip Connection - [N, F_l, H, W]
        """
        N, _, H, W = x.size()
        N, _, H_g, W_g = g.size()

        # 1x1 Conv로 g와 x의 채널 정렬
        g_1: torch.Tensor = self.W_g(g)  # [N, F_int, H, W]
        x_1: torch.Tensor = self.W_x(x)  # [N, F_int, H, W]

        # Flatten H and W into one dimension (to match MHA input requirements)
        g_proj_flat = g_1.flatten(2).permute(0, 2, 1)  # [N, H*W, F_int]
        x_proj_flat = x_1.flatten(2).permute(0, 2, 1)  # [N, H*W, F_int]

        if getattr(self, 'positional_encoding', None):
            g_proj_flat = self.positional_encoding(g_proj_flat)
            x_proj_flat = self.positional_encoding(x_proj_flat)

        # Multihead Attention
        attention_features, _ = self.attention(
            query=g_proj_flat, key=x_proj_flat, value=x_proj_flat
        )  # [N, H*W, F_int]

        # Reshape back to spatial dimensions
        attention_features = attention_features.permute(0, 2, 1).view(N, -1, H_g, W_g)  # [N, F_int, H, W]

        # Attention Map 계산
        alpha = self.psi(attention_features)

        # Combine attention output with skip connection
        alpha = self.sigmoid(alpha)

        alpha = F.interpolate(alpha, size=x.size()[2:], mode='bilinear', align_corners=True)  # [N, 1, H_x, W_x]

        # Attention Map과 Skip Connection의 요소별 곱
        out = alpha * x  # [N, F_l, H, W]

        return out
    

class MHAUNet(nn.Module):

    def __init__(
            self,
            in_channels=3,
            n_classes=1,
            is_deconv=True,
            is_batchnorm=True,
            attn_F_int=16,
            attn_num_heads=4,
            attn_positional_encoding=True,
            input_size=(512, 512),
        ):
        super(MHAUNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.n_classes = n_classes
        #
        # filters = [32, 64, 128, 256, 512]
        self.filters = filters = [64, 128, 256, 512, 1024]
        # # filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # max_len calculation
        max_len = (input_size[0] // 2**3) * (input_size[1] // 2**3)
        # attention layers
        self.attn_1 = MHABlock(
            F_g=filters[4], F_l=filters[3], F_int=attn_F_int, num_heads=attn_num_heads,
            positional_encoding=attn_positional_encoding, max_len=max_len
        )
        max_len = (input_size[0] // 2**2) * (input_size[1] // 2**2)
        self.attn_2 = MHABlock(
            F_g=filters[3], F_l=filters[2], F_int=attn_F_int, num_heads=attn_num_heads,
            positional_encoding=attn_positional_encoding, max_len=max_len
        )
        max_len = (input_size[0] // 2**1) * (input_size[1] // 2**1)
        self.attn_3 = MHABlock(
            F_g=filters[2], F_l=filters[1], F_int=attn_F_int, num_heads=attn_num_heads,
            positional_encoding=attn_positional_encoding, max_len=max_len
        )
        max_len = (input_size[0]) * (input_size[1])
        self.attn_4 = MHABlock(
            F_g=filters[1], F_l=filters[0], F_int=attn_F_int, num_heads=attn_num_heads,
            positional_encoding=attn_positional_encoding, max_len=max_len
        )

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        #
        self.outconv1 = nn.Conv2d(filters[0], self.n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def dotProduct(self,seg,cls):
        B, N, H, W = seg.size()
        seg = seg.view(B, N, H * W)
        final = torch.einsum("ijk,ij->ijk", [seg, cls])
        final = final.view(B, N, H, W)
        return final

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, tuple[torch.Tensor, ...]]]:
        '''
        Args:
            inputs (torch.Tensor): input image tensor
        Returns:
            tuple[torch.Tensor, tuple[torch.Tensor, tuple[torch.Tensor, ...]]]: segmentation map, (embedding, (attention maps for each layer))
        '''
        # print("inputs", inputs.shape)           # 3*512*512
        conv1 = self.conv1(inputs)
        # print("conv1", conv1.shape)             # 64*512*512
        maxpool1 = self.maxpool1(conv1)
        # print("maxpool1", maxpool1.shape)       # 64*256*256


        conv2 = self.conv2(maxpool1)
        # print("conv2", conv2.shape)             # 128*256*256
        maxpool2 = self.maxpool2(conv2)
        # print("maxpool2", maxpool2.shape)       # 128*128*128


        conv3 = self.conv3(maxpool2)
        # print("conv3", conv3.shape)             # 256*128*128
        maxpool3 = self.maxpool3(conv3)
        # print("maxpool3", maxpool3.shape)       # 256*64*64


        conv4 = self.conv4(maxpool3)
        # print("conv4", conv4.shape)             # 512*64*64
        maxpool4 = self.maxpool4(conv4)
        # print("maxpool4", maxpool4.shape)       # 512*32*32

        center = self.center(maxpool4)
        # print("center", center.shape)           # 1024*32*32

        # conv4 = (B, 512, 64, 64)
        attn_1 = self.attn_1(center, conv4)
        # print("attn_1", attn_1.shape)                         # 512*64*64
        up4 = self.up_concat4(center, attn_1)
        # print("up4", up4.shape)                               # 512*64*64

        attn_2 = self.attn_2(up4, conv3)
        # print("attn_2", attn_2.shape)                         # 256*128*128
        up3 = self.up_concat3(up4, attn_2)
        # print("up3", up3.shape)                               # 256*128*128

        attn_3 = self.attn_3(up3, conv2)
        # print("attn_3", attn_3.shape)                         # 128*256*256
        up2 = self.up_concat2(up3, attn_3)
        # print("up2", up2.shape)                               # 128*256*256

        attn_4 = self.attn_4(up2, conv1)
        # print("attn_4", attn_4.shape)                         # 64*512*512
        up1 = self.up_concat1(up2, attn_4)
        # print("up1", up1.shape)                               # 64*512*512

        d1 = self.outconv1(up1)
        # print("d1", d1.shape)                               # 1*512*512

        return d1, (up1, (attn_1, attn_2, attn_3, attn_4))
