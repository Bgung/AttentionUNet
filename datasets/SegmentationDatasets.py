import os
import cv2
import json
import torch

import numpy as np
import albumentations as A

from PIL import Image
from torchvision import transforms as TorchTransforms
from albumentations.pytorch import ToTensorV2

from config import Config
from .BaseDataset import BaseDataset

class MedicalDataset(BaseDataset):

    @property
    def path(self) -> str:
        raise NotImplementedError
    
    @property
    def dataset_root(self) -> str:
        return os.path.join(self.root, self.path)

    @property
    def dataset_name(self) -> str:
        raise NotImplementedError

    @property
    def num_classes(self) -> int:
        if self.__class__.__name__ in [
            'CHASEDB1', 'EBHI', 'HRF', 'KVASIR'
        ]:
            return 1
        else:
            raise NotImplementedError

    @property
    def is_train(self) -> bool:
        return False


    def apply_transforms(self, img: np.ndarray, ann: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        augmented = self.base_transforms(image=img, mask=ann)
        img: np.ndarray = augmented['image']
        ann: np.ndarray = augmented['mask']
        if self.transforms and self.is_train:
            augmented = self.transforms(image=img, mask=ann)
            img: np.ndarray = augmented['image']
            ann: np.ndarray = augmented['mask']
        
        return TorchTransforms.ToTensor()(img), TorchTransforms.ToTensor()(ann)
    

    def __init__(
            self,
            config: Config,
        ) -> None:
        super().__init__(
            config.dataset.root
        )
        
        self.base_transforms = A.Compose([
            A.Resize(*config.dataset.input_shape),
        ])
        if config.dataset.transforms is None:
            transforms = []
        else:
            if isinstance(config.dataset.transforms, list):
                transforms = config.dataset.transforms
            else:
                transforms = config.dataset.transforms.transforms

        self.transforms = A.Compose(transforms)
        

class CHASEDB1(MedicalDataset):

    @property
    def path(self) -> str:
        return 'CHASEDB1'

    @property
    def dataset_name(self) -> str:
        return 'CHASEDB1'

    CLS_MAPPER = {
        'bg': 0,
        'vessel': 1
    }
    CLS_MAPPER_INV = {v: k for k, v in CLS_MAPPER.items()}
    
    def __init__(
            self, 
            config: Config,
        ) -> None:
        '''
        Args:
            root: root directory of the dataset
            transforms: transforms to apply to the images. ToTensor() is always applied
        '''
        super().__init__(config)
        imgs_anns = os.listdir(os.path.join(self.dataset_root))

        self.img_L = [f for f in imgs_anns if f.endswith('L.jpg')]
        self.img_R = [f for f in imgs_anns if f.endswith('R.jpg')]

        self.anns_L_1st = [f for f in imgs_anns if f.endswith('L_1stHO.png')]
        self.anns_R_1st = [f for f in imgs_anns if f.endswith('R_1stHO.png')]
        self.anns_L_2nd = [f for f in imgs_anns if f.endswith('L_2ndHO.png')]
        self.anns_R_2nd = [f for f in imgs_anns if f.endswith('R_2ndHO.png')]

        self.img = self.img_L + self.img_R
        self.anns_1st = self.anns_L_1st + self.anns_R_1st
        self.anns_2nd = self.anns_L_2nd + self.anns_R_2nd
            

    def __getitem__(self, idx: int) -> dict[str, Image.Image]:
        '''
        img: (3, H, W)
        ann: (1, H, W)
        '''
        img = self.get_image(idx)
        ann: list[Image.Image] = self.get_annotation(idx)

        img = np.array(img[0])

        ann[0], ann[1] = np.array(ann[0]), np.array(ann[1])
        # intersection of 1stHO and 2ndHO
        ann = np.logical_and(ann[0], ann[1])
        ann = ann.astype(np.int8)

        img, ann = self.apply_transforms(img, ann)

        return {'img': img, 'ann': ann}


    def __len__(self) -> int:
        return len(self.img)
    

    def get_image(self, idx: int) -> list[Image.Image]:
        path = [os.path.join(self.dataset_root, self.img[idx])]
        return [Image.open(p) for p in path]

    
    def get_annotation(self, idx: int) -> list[str]:
        path = [os.path.join(self.dataset_root, self.anns_1st[idx]), os.path.join(self.dataset_root, self.anns_2nd[idx])]
        return [Image.open(p) for p in path]


class EBHI(MedicalDataset):

    @property
    def path(self) -> str:
        return 'EBHI-SEG'

    @property
    def dataset_name(self) -> str:
        return 'EBHI'
    
    CLS_MAPPER = {
        'bg': 0,
        'Normal': 1,
        'Adenocarcinoma': 2,
        'HighGradeIN': 3,
        'LowGradeIN': 4,
        'Polyp': 5,
        'SerratedAdenoma': 6,
    }
    CLS_MAPPER_INV = {v: k for k, v in CLS_MAPPER.items()}
    
    def __init__(
            self,
            config: Config,
        ) -> None:
        super().__init__(config)
        self.build_dataset()

    def __getitem__(self, idx):
        data = self.data[idx]

        img = np.array(Image.open(data['img']))
        ann = np.array(Image.open(data['ann']))
        cls = self.CLS_MAPPER[data['cls']]

        img, ann = self.apply_transforms(img, ann)

        return {'img': img, 'ann': ann, 'cls': cls}
    
    def build_dataset(self):
        cls_dirs = os.listdir(self.dataset_root)
        cls_dirs = [
            os.path.join(self.dataset_root, d) for d in cls_dirs 
            if os.path.isdir(os.path.join(self.dataset_root, d))
        ]

        self.classes = len(cls_dirs)

        self.data: list[dict[
            str, str, # Key: img Value: path
            str, str, # Key: ann Value: path
            str, str  # Key: cls Value: class idx
        ]] = []
        for idx, cls_dir in enumerate(cls_dirs):
            cls = self.CLS_MAPPER_INV[idx]

            img_dir = os.path.join(cls_dir, 'image')
            ann_dir = os.path.join(cls_dir, 'label')

            imgs = os.listdir(img_dir)
            anns = os.listdir(ann_dir)

            imgs = [os.path.join(img_dir, i) for i in imgs]
            anns = [os.path.join(ann_dir, a) for a in anns]

            for img, ann in zip(imgs, anns):
                self.data.append({
                    'img': img,
                    'ann': ann,
                    'cls': cls
                })

    def __len__(self):
        return len(self.data)

class HRF(MedicalDataset):

    @property
    def path(self) -> str:
        return 'High-Resolution Fundus (HRF) Image Database'

    @property
    def dataset_name(self) -> str:
        return 'HRF'
    
    CLS_MAPPER = {
        'bg': 0,    # Background
        'h': 1,     # Healthy vessel
        'g': 2,     # Glaucomatous vessel
        'dr': 3,    # Diabetic Retinopathy vessel
    }
    CLS_MAPPER_INV = {v: k for k, v in CLS_MAPPER.items()}
    
    def __init__(
            self,
            config: Config,
        ) -> None:
        super().__init__(config)
        self.build_dataset()

    def __getitem__(self, idx: int) -> dict[str, Image.Image]:
        data = self.data[idx]

        img = np.array(Image.open(data['img']))
        ann = np.array(Image.open(data['ann']))
        label = data['cls']

        img, ann = self.apply_transforms(img, ann)

        return {'img': img, 'ann': ann, 'cls': label}

    def __len__(self) -> int:
        return len(self.data)

    def build_dataset(self):
        images = os.listdir(os.path.join(self.dataset_root, 'images'))
        masks = os.listdir(os.path.join(self.dataset_root, 'manual1'))

        self.data: list[dict[
            str, str, # Key: img Value: path
            str, str,  # Key: ann Value: path
            str, str  # Key: cls Value: class idx
        ]] = []
        for img, mask in zip(images, masks):
            self.data.append({
                'img': os.path.join(self.dataset_root, 'images', img),
                'ann': os.path.join(self.dataset_root, 'manual1', mask),
                'cls': self.CLS_MAPPER[img.split('_')[1].split('.')[0]]
            })


class KVASIR(MedicalDataset):

    @property
    def path(self) -> str:
        return 'kvasir-seg'
    
    @property
    def dataset_name(self) -> str:
        return 'KVASIR'

    CLS_MAPPER = {
        'bg': 0,    # Background
        'polyp': 1, # Polyp
    }
    CLS_MAPPER_INV = {v: k for k, v in CLS_MAPPER.items()}

    def __init__(
            self,
            config: Config,
        ) -> None:
        super().__init__(config)
        self.metadata = json.load(
            open(os.path.join(self.dataset_root, 'kavsir_bboxes.json'))
        )
        self.build_dataset()

    def __getitem__(self, idx: int) -> dict[str, Image.Image]:
        data = self.data[idx]

        img = np.array(Image.open(data['img']))
        ann = np.array(Image.open(data['ann']))


        img, ann = self.apply_transforms(img, ann)

        # ann: (3, H, W) => (1, H, W)
        ann = ann[0].unsqueeze(0)  # ann[0] == ann[1] == ann[2]

        return {'img': img, 'ann': ann}

    def __len__(self) -> int:
        return len(self.data)

    def build_dataset(self):
        self.data: list[dict[
            str, str, # Key: img Value: path
            str, str,  # Key: ann Value: path
            str, str  # Key: cls Value: class idx
        ]] = []

        for key, value in self.metadata.items():
            bboxes = value['bbox']

            label = bboxes[0]['label']

            self.data.append({
                'img': os.path.join(self.dataset_root, 'images', key) + '.jpg',
                'ann': os.path.join(self.dataset_root, 'masks', key) + '.jpg',
            })