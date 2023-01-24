import os

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class ArtDataset(Dataset):
    def __init__(self, dir_path: str, augment: bool = True, img_size: int = 256):
        super(ArtDataset, self).__init__()
        self.root_dir = dir_path
        self.path_to_grayscale = self.root_dir + "_black"
        self.path_to_color = self.root_dir + "_color"
        self.list_files_grayscale = os.listdir(self.path_to_grayscale)
        self.list_files_color = os.listdir(self.path_to_color)
        self.img_size = img_size
        self.augment = augment
        self.resize_augment = A.Compose(
            [A.Resize(width=self.img_size, height=self.img_size)]
        )
        self.input_augment_advance = A.Compose(
            [
                A.RandomBrightnessContrast(p=0.1),
                A.ImageCompression(quality_lower=90, quality_upper=100, p=0.1),
                A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0, 2), p=0.1),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )
        self.input_augment_simple = A.Compose(
            [
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )
        self.target_augment_simple = A.Compose(
            [
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )

    def __len__(self):
        return len(self.list_files_grayscale)

    def __getitem__(self, idx):
        grayscale_file = self.list_files_grayscale[idx]
        color_file = self.list_files_color[idx]
        grayscale_img_path = os.path.join(self.path_to_grayscale, grayscale_file)
        color_img_path = os.path.join(self.path_to_color, color_file)
        grayscale_img = np.array(Image.open(grayscale_img_path))
        color_img = np.array(Image.open(color_img_path))
        grayscale_img, color_img = self.resize_augment(image=grayscale_img)["image"], self.resize_augment(image=color_img)["image"]

        if self.augment:
            input_image = self.input_augment_advance(image=grayscale_img)
        else:
            input_image = self.input_augment_simple(image=grayscale_img)
        target_image = self.target_augment_simple(image=color_img)

        return input_image["image"], target_image["image"]# , positive, negative
