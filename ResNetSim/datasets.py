import os
import random

import cv2
import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageNet

class FlowersDataset(Dataset):
    def __init__(self, dir_path: str, augment: bool = True, img_size: int = 256):
        super(FlowersDataset, self).__init__()
        self.root_dir = dir_path
        self.path_to_images = self.root_dir
        self.list_files = [n for n in os.listdir(self.path_to_images) if n.endswith('.png')]
        self.img_size = img_size
        self.augment = augment
        self.input_augment_advance = A.Compose(
            [
                A.Resize(width=self.img_size, height=self.img_size, interpolation=cv2.INTER_LINEAR),
                A.RandomBrightnessContrast(p=0.1),
                A.ImageCompression(quality_lower=90, quality_upper=100, p=0.1),
                A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0, 2), p=0.1),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )
        self.input_augment_simple = A.Compose(
            [
                A.Resize(width=self.img_size, height=self.img_size, interpolation=cv2.INTER_LINEAR),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, idx):
        file = self.list_files[idx]
        img_path = os.path.join(self.path_to_images, file)
        img = np.array(Image.open(img_path).convert('RGB'))
        if self.augment:
            input_image = self.input_augment_advance(image=img)
        else:
            input_image = self.input_augment_simple(image=img)


        return input_image["image"]

