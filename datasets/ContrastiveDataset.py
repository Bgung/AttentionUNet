import os
import torch

import numpy as np

from PIL import Image

from torchvision import transforms as TorchTransforms

from .BaseDataset import BaseDataset

'''
대조 학습을 위한 데이터셋
출력: (img1, img2, label)
    img1: 
    img2: 
'''
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torchvision import transforms as TorchTransforms


class MedicalDataset(Dataset):
    def __init__(
            self, root: str,
            transforms: TorchTransforms = None
        ) -> None:
        super().__init__()
        self.root = root
        
        if transforms is None:
            transforms = TorchTransforms.Compose([
                TorchTransforms.ToTensor(),
                TorchTransforms.Resize((512, 512))
            ])
        self.transforms = transforms

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx: int):
        raise NotImplementedError


class CHASEDB1(MedicalDataset):
    def __init__(
            self, root: str,
            transforms: TorchTransforms = None,
            num_pairs: int = 10  # Number of positive and negative pairs
        ) -> None:
        '''
        Args:
            root: root directory of the dataset
            transforms: transforms to apply to the images. ToTensor() is always applied
            num_pairs: number of positive/negative pairs to generate per image
        '''
        super().__init__(root=root, transforms=transforms)
        self.num_pairs = num_pairs

        imgs_anns = os.listdir(os.path.join(self.root))

        self.img_L = [f for f in imgs_anns if f.endswith('L.jpg')]
        self.img_R = [f for f in imgs_anns if f.endswith('R.jpg')]

        self.anns_L_1st = [f for f in imgs_anns if f.endswith('L_1stHO.png')]
        self.anns_R_1st = [f for f in imgs_anns if f.endswith('R_1stHO.png')]
        self.anns_L_2nd = [f for f in imgs_anns if f.endswith('L_2ndHO.png')]
        self.anns_R_2nd = [f for f in imgs_anns if f.endswith('R_2ndHO.png')]

        self.img = self.img_L + self.img_R
        self.anns_1st = self.anns_L_1st + self.anns_R_1st
        self.anns_2nd = self.anns_L_2nd + self.anns_R_2nd

    def __getitem__(self, idx: int) -> dict:
        '''
        Args:
            idx: index of the image
        Returns:
            A dictionary containing:
                - img: preprocessed input image (C, H, W)
                - ann: binary segmentation mask (1, H, W)
                - positive_pairs: list of positive pixel pairs (same class)
                - negative_pairs: list of negative pixel pairs (different classes)
        '''
        img = self.get_image(idx)
        ann: list[Image.Image] = self.get_annotation(idx)

        img = np.array(img[0])

        # Combine annotations
        ann[0], ann[1] = np.array(ann[0]), np.array(ann[1])
        ann = np.logical_and(ann[0], ann[1])
        ann = ann.astype(np.int8)

        if self.transforms:
            img: torch.Tensor = self.transforms(img)
            ann: torch.Tensor = self.transforms(ann)

        # Generate positive and negative pairs
        positive_pairs, negative_pairs = self.generate_pixel_pairs(ann)

        return {
            'img': img,
            'ann': ann,
            'positive_pairs': positive_pairs,
            'negative_pairs': negative_pairs
        }

    def __len__(self) -> int:
        return len(self.img)

    def get_image(self, idx: int) -> list[Image.Image]:
        path = [os.path.join(self.root, self.img[idx])]
        return [Image.open(p) for p in path]

    def get_annotation(self, idx: int) -> list[Image.Image]:
        path = [os.path.join(self.root, self.anns_1st[idx]), os.path.join(self.root, self.anns_2nd[idx])]
        return [Image.open(p) for p in path]

    def generate_pixel_pairs(self, ann: torch.Tensor) -> tuple:
        """
        Generate positive and negative pixel pairs.
        Args:
            ann: segmentation mask (1, H, W)
        Returns:
            positive_pairs: list of positive pixel pairs
            negative_pairs: list of negative pixel pairs
        """
        ann = ann.squeeze(0)  # Remove channel dimension
        h, w = ann.shape

        # Flatten annotation for indexing
        ann_flat = ann.view(-1)
        positive_pairs = []
        negative_pairs = []

        # Unique classes in the annotation
        unique_classes = torch.unique(ann_flat)

        # Generate positive pairs
        for cls in unique_classes:
            if cls == 0:  # Ignore background
                continue
            class_indices = (ann_flat == cls).nonzero(as_tuple=False).squeeze()
            if len(class_indices) > 1:
                sampled_indices = class_indices[torch.randperm(len(class_indices))[:self.num_pairs]]
                for i in range(len(sampled_indices) - 1):
                    for j in range(i + 1, len(sampled_indices)):
                        positive_pairs.append((sampled_indices[i].item(), sampled_indices[j].item()))

        # Generate negative pairs
        for cls in unique_classes:
            if cls == 0:
                continue
            class_indices = (ann_flat == cls).nonzero(as_tuple=False).squeeze()
            negative_indices = (ann_flat != cls).nonzero(as_tuple=False).squeeze()
            if len(class_indices) > 0 and len(negative_indices) > 0:
                sampled_negatives = negative_indices[torch.randperm(len(negative_indices))[:self.num_pairs]]
                sampled_anchors = class_indices[torch.randperm(len(class_indices))[:self.num_pairs]]
                for anchor, neg in zip(sampled_anchors, sampled_negatives):
                    negative_pairs.append((anchor.item(), neg.item()))

        return positive_pairs, negative_pairs


class EBHI(MedicalDataset):
    CLS_MAPPER = {
        'Normal': 0,
        'Adenocarcinoma': 1,
        'HighGradeIN': 2,
        'LowGradeIN': 3,
        'Polyp': 4,
        'SerratedAdenoma': 5,
    }
    CLS_MAPPER_INV = {v: k for k, v in CLS_MAPPER.items()}
    
    def __init__(
            self, root: str,
            transorms: TorchTransforms=None
        ) -> None:
        super().__init__(root=root, transforms=transorms)

        self.build_dataset()

    def __getitem__(self, idx):
        print(self.data)
        print(type(self.data))
        data = self.data[idx]

        img = np.array(Image.open(data['img']))
        ann = np.array(Image.open(data['ann']))
        cls = self.CLS_MAPPER[data['cls']]

        print(ann)
        print(ann.max(), ann.min())

        if self.transforms:
            img = self.transforms(img)
            ann = self.transforms(ann)

        return {'img': img, 'ann': ann, 'cls': cls}
    
    def build_dataset(self):
        cls_dirs = os.listdir(self.root)
        cls_dirs = [os.path.join(self.root, d) for d in cls_dirs if os.path.isdir(d)]

        self.classes = len(cls_dirs)

        self.data: list[dict[
            str, str, # Key: img Value: path
            str, str, # Key: ann Value: path
            str, str  # Key: cls Value: class name
        ]] = []
        for idx, cls_dir in enumerate(cls_dirs):
            cls = self.CLS_MAPPER_INV[idx]

            img_dir = os.path.join(cls_dir, 'image')
            ann_dir = os.path.join(cls_dir, 'label')

            imgs = os.listdir(img_dir)
            anns = os.listdir(ann_dir)

            imgs = [os.path.join(img_dir, i) for i in imgs]
            anns = [os.path.join(ann_dir, a) for a in anns]

            print(imgs)

            for img, ann in zip(imgs, anns):
                self.data.append({
                    'img': img,
                    'ann': ann,
                    'cls': cls
                })

    def __len__(self):
        return len(self.data)

class HRF(BaseDataset):
    ...

class KVASIR(BaseDataset):
    ...
