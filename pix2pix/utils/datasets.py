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
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )
        self.input_augment_simple = A.Compose(
            [
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )
        self.target_augment_simple = A.Compose(
            [
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )

    def __len__(self):
        return len(self.list_files_grayscale)

    def __getitem__(self, idx):
        rand_idx = idx
        while rand_idx == idx:
            rand_idx = random.randint(0, self.__len__()-1)
        grayscale_file = self.list_files_grayscale[idx]
        color_file = self.list_files_color[idx]
        random_file = self.list_files_grayscale[rand_idx]
        grayscale_img_path = os.path.join(self.path_to_grayscale, grayscale_file)
        color_img_path = os.path.join(self.path_to_color, color_file)
        random_img_path = os.path.join(self.path_to_color, random_file)
        grayscale_img = np.array(Image.open(grayscale_img_path))
        color_img = np.array(Image.open(color_img_path))
        random_img = np.array(Image.open(random_img_path))
        positive_img = np.round(grayscale_img*.8 + random_img*.2).astype(np.uint8)
        negative_img = random_img
        grayscale_img, color_img, positive_img, negative_img = self.resize_augment(image=grayscale_img)["image"], self.resize_augment(image=color_img)["image"], \
            self.resize_augment(image=positive_img)["image"], self.resize_augment(image=negative_img)["image"]

        if self.augment:
            input_image = self.input_augment_advance(image=grayscale_img)
        else:
            input_image = self.input_augment_simple(image=grayscale_img)
        target_image = self.target_augment_simple(image=color_img)
        positive_image = self.target_augment_simple(image=positive_img)
        negative_image = self.target_augment_simple(image=negative_img)

        return input_image["image"].to(torch.float32), target_image["image"].to(torch.float32), positive_image["image"].to(torch.float32), negative_image["image"].to(torch.float32)


class ImageNetCustomDataset(Dataset):
    def __init__(self, augment: bool = True, img_size: int = 256):
        super(ImageNetCustomDataset, self).__init__()
        self.imagenet = ImageNet()
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
        return self.imagenet.__len__()

    def __getitem__(self, idx):
        image, trg_class = self.imagenet(idx)
        rand_idx = idx
        while rand_idx == idx:
            rand_idx = random.randint(0, self.imagenet.__len__()-1)
        grayscale_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        color_img = image
        random_img = self.imagenet(rand_idx)
        positive_img = np.round(color_img * .8 + random_img * .2).astype(np.uint8)
        negative_img = random_img
        grayscale_img, color_img, positive_img, negative_img = self.resize_augment(image=grayscale_img)["image"], \
        self.resize_augment(image=color_img)["image"], \
            self.resize_augment(image=positive_img)["image"], self.resize_augment(image=negative_img)["image"]

        if self.augment:
            input_image = self.input_augment_advance(image=grayscale_img)
        else:
            input_image = self.input_augment_simple(image=grayscale_img)
        target_image = self.target_augment_simple(image=color_img)
        positive_image = self.target_augment_simple(image=positive_img)
        negative_image = self.target_augment_simple(image=negative_img)

        return input_image["image"], target_image["image"], positive_image["image"], negative_image["image"]



if __name__ == '__main__':
    dataset = ImageNetCustomDataset()
    loader = DataLoader(dataset=dataset, batch_size=32, num_workers=8, shuffle=True)
    batch = next(iter(loader))



