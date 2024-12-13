# import os
# import cv2
# import torch

# import numpy as np
# import albumentations as A

# from torchvision import transforms as TorchTransforms
# from PIL import Image

# from config import Config
# from datasets.SegmentationDatasets import CHASEDB1, EBHI, HRF, KVASIR

# config = Config()
# config.dataset.input_shape = (512, 512)
# config.dataset.transforms = TorchTransforms.Compose([
#     A.HorizontalFlip(p=0.5),
#     A.VerticalFlip(p=0.5),
#     A.Rotate(limit=30, p=0.5),
#     A.Affine(scale=(0.9, 1.1), translate_percent=(0.1, 0.1), shear=10, p=0.5),
#     A.Normalize(mean=(0.5,), std=(0.5,)),  # Normalize image
# ])
# chasedb = CHASEDB1(config)

# data = chasedb[0]
# img = data['img']
# ann = data['ann']

# n_img = torch.zeros((3, 512, 1024))
# n_img[:, :, :512] = img
# n_img[:, :, 512:] = ann

# n_img = n_img.permute(1, 2, 0).numpy()
# n_img = (n_img * 255).astype(np.uint8)
# n_img = Image.fromarray(n_img)
# n_img.save("augmented.png")

import torch
from models.implemented.UNet import MHAUNet, AttentionUNet

attenunet = AttentionUNet(in_channels=3, n_classes=1)
mhaunet = MHAUNet(in_channels=3, n_classes=1)

input = torch.randn((1, 3, 64, 64))
print("-------Attention UNet-------")
output = attenunet(input)
# print(output[0].shape) # segmentation mask
# print(output[1].shape) # 
print("-------MHA UNet-------")
output = mhaunet(input)
# print(output[0].shape) # segmentation mask
# print(output[1].shape) #

# num_classes = 3
# logits = torch.randn((4, num_classes, 512, 512)).cuda()  # Model output
# ground_truth = torch.randint(0, num_classes, (4, 512, 512)).cuda()  # Ground truth mask
# print(logits.shape, ground_truth.shape)
# print(logits.dtype, ground_truth.dtype)
# criterion = torch.nn.CrossEntropyLoss()
# loss = criterion(logits, ground_truth)
# print(loss)

# from torchmetrics.segmentation import MeanIoU
# from torchmetrics.segmentation import GeneralizedDiceScore

# miou = MeanIoU(num_classes=num_classes)
# dice = GeneralizedDiceScore(num_classes=num_classes)

# preds = torch.randint(0, num_classes, (10, 512, 512)) # B, C, H, W
# target = torch.randint(0, num_classes, (10, 512, 512)) # B, C, H, W
# # target = torch.nn.functional.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
# print(preds.shape, target.shape)
# print(miou(preds, target))
# print(dice(preds, target))


# chase = CHASEDB1("data/CHASEDB1")
# print(chase[0])
# print(chase[0]['img'].shape)
# print(chase[0]['ann'].shape)

# ebhi = EBHI("data/EBHI-SEG")
# # print(ebhi[0])
# print(ebhi[0]['img'].shape)
# print(ebhi[0]['ann'].shape)
# print(ebhi[0]['ann'].unique())

# hrf = HRF("data/High-Resolution Fundus (HRF) Image Database")
# # print(hrf[0])
# print(hrf[0]['img'].shape)
# print(hrf[0]['ann'].shape)
# print(hrf[0]['ann'].unique())


# kvasir = KVASIR("data/kvasir-seg")
# print(kvasir[0])
# print(kvasir[0]['img'].shape)
# print(kvasir[0]['ann'].shape)
# print(kvasir[0]['ann'].unique())

