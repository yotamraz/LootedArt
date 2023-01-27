import os
import sys
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms
from torch import optim
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError, StructuralSimilarityIndexMeasure
from tqdm import tqdm

from config import Config
from models.discriminator import Discriminator
from models.generator import Generator
from utils.datasets import ArtDataset
from utils.general import save_checkpoint
from utils.loss import *


class Trainer:
    def __init__(self, opt):
        self.config = Config()
        self.opt = opt

        self.discriminator = Discriminator(in_channels=3).to(device=self.opt.device)
        self.generator = Generator(in_channels=self.config.input_channels, out_channels=3, features=64).to(
            device=self.opt.device)

        self.opt_disc = optim.Adam(self.discriminator.parameters(), lr=self.config.lr, betas=(0.5, 0.999))
        self.opt_gen = optim.Adam(self.generator.parameters(), lr=self.config.lr, betas=(0.5, 0.999))

        self.g_scaler = torch.cuda.amp.GradScaler()
        self.d_scaler = torch.cuda.amp.GradScaler()

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        self.triplet_loss = nn.TripletMarginWithDistanceLoss(
            margin=2.0,
            distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y)
        )
        self.perceptual_loss = VGGLoss()
        self.tv_loss = TVLoss(p=2)
        self.mse = MeanSquaredError().to(self.opt.device)
        self.ssim = StructuralSimilarityIndexMeasure().to(self.opt.device)

        self.train_dataset = ArtDataset(dir_path=os.path.join(self.opt.data_path, "train"), augment=True,
                                        img_size=self.opt.img_size)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.opt.batch_size, shuffle=True,
                                       num_workers=self.config.num_workers)

        self.val_dataset = ArtDataset(dir_path=os.path.join(self.opt.data_path, "test"), augment=False,
                                      img_size=self.opt.img_size)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.opt.batch_size, shuffle=True,
                                     num_workers=self.config.num_workers)

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            torchvision.transforms.GaussianBlur(kernel_size=[5, 5]),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomAdjustSharpness(sharpness_factor=0.1),
            torchvision.transforms.RandomAffine(degrees=0.2),
            torchvision.transforms.RandomRotation(degrees=0.3),
        ])

    def _apply_transformations(self, tensor):
        return self.transform(tensor)

    def _shuffle_tensor(self, tensor):
        idx = torch.randperm(tensor.shape[0])
        return tensor[idx].view(tensor.size())

    def train_loop(self):
        for epoch in range(self.config.num_epochs):
            self.train_step(epoch)
            self.val_step(epoch)

            if self.config.save_model and epoch % self.config.save_period == 0:
                save_checkpoint(self.discriminator, optimizer=self.opt_disc,
                                filename=f"./checkpoints/discriminator_{epoch}.pth")
                save_checkpoint(self.generator, optimizer=self.opt_gen, filename=f"./checkpoints/generator_{epoch}.pth")

    def train_step(self, epoch):
        loop = tqdm(self.train_loader, leave=True)
        self.discriminator.train(), self.generator.train()
        for idx, (inputs, targets, positives, negatives) in enumerate(loop):
            inputs, targets, positives, negatives = inputs.to(self.opt.device), targets.to(
                self.opt.device), positives.to(self.opt.device), negatives.to(self.opt.device)
            self.generator.zero_grad()
            self.discriminator.zero_grad()
            # train discriminator
            latent_positives, _ = self.generator(positives)
            latent_negatives, _ = self.generator(negatives)
            latent_inputs, target_fake = self.generator(inputs)
            d_real = self.discriminator(inputs, targets)
            d_fake = self.discriminator(inputs, target_fake.detach())
            d_real_loss = self.bce_loss(d_real, torch.ones_like(d_real))
            d_fake_loss = self.bce_loss(d_fake, torch.zeros_like(d_fake))
            d_loss = (d_real_loss + d_fake_loss) / 2
            # backprop
            self.opt_disc.zero_grad()
            self.d_scaler.scale(d_loss).backward()
            self.d_scaler.step(self.opt_disc)
            self.d_scaler.update()

            # train generator
            self.discriminator.zero_grad()
            d_fake = self.discriminator(inputs, target_fake)
            g_fake_loss = self.bce_loss(d_fake, torch.ones_like(d_fake))
            l1_loss = self.l1_loss(target_fake, targets) * self.config.l1_lambda
            triplet_loss = self.triplet_loss(latent_inputs, latent_positives,
                                             latent_negatives) * self.config.triplet_lambda
            g_loss = g_fake_loss + l1_loss + triplet_loss
            ssim = self.ssim(target_fake, targets)
            # backprop
            self.opt_gen.zero_grad()
            self.g_scaler.scale(g_loss).backward()
            self.g_scaler.step(self.opt_gen)
            self.g_scaler.update()

            if idx % 5 == 0:
                loop.set_postfix(
                    epoch=f"{epoch + 1}/{self.config.num_epochs}",
                    subset="train",
                    g_loss=g_loss.mean().item(),
                    d_loss=d_loss.mean().item(),
                    accuracy_ssim=ssim.mean().item()
                )

    def val_step(self, epoch):
        loop = tqdm(self.val_loader, leave=True)
        self.discriminator.eval(), self.generator.eval()
        for idx, (inputs, targets, positives, negatives) in enumerate(loop):
            inputs, targets, positives, negatives = inputs.to(self.opt.device), targets.to(
                self.opt.device), positives.to(self.opt.device), negatives.to(self.opt.device)
            with torch.no_grad():

                # eval discriminator
                latent_inputs, target_fake = self.generator(inputs)
                latent_positives, _ = self.generator(positives)
                latent_negatives, _ = self.generator(negatives)
                d_real = self.discriminator(inputs, targets)
                d_fake = self.discriminator(inputs, target_fake.detach())
                d_real_loss = self.bce_loss(d_real, torch.ones_like(d_real))
                d_fake_loss = self.bce_loss(d_fake, torch.zeros_like(d_fake))
                d_loss = (d_real_loss + d_fake_loss) / 2

                # eval generator
                d_fake = self.discriminator(inputs, target_fake)
                g_fake_loss = self.bce_loss(d_fake, torch.ones_like(d_fake))
                l1_loss = self.l1_loss(target_fake, targets) * self.config.l1_lambda
                triplet_loss = self.triplet_loss(latent_inputs, latent_positives,
                                                 latent_negatives) * self.config.triplet_lambda
                g_loss = g_fake_loss + l1_loss + triplet_loss
                ssim = self.ssim(target_fake, targets)

                if idx % 5 == 0:
                    loop.set_postfix(
                        epoch=f"{epoch + 1}/{self.config.num_epochs}",
                        subset="val",
                        g_loss=g_loss.mean().item(),
                        d_loss=d_loss.mean().item(),
                        accuracy_ssim=ssim.mean().item()
                    )


def train():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./sample_data", help="path to dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="mini batch size for training")
    parser.add_argument("--img_size", type=int, default=256, help="size of input and target images")
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:0/1/2/etc for GPU -1 for CPU")
    opt = parser.parse_args()

    trainer = Trainer(opt)
    trainer.train_loop()


if __name__ == "__main__":
    train()
