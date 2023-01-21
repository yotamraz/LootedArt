import argparse
import os

import torchvision
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.generator import Generator
from utils.general import load_checkpoint
from utils.datasets import ArtDataset
from config import Config


class Inference:
    def __init__(self, opt, examples=10):
        self.config = Config()
        self.data_path = opt.data_path
        self.batch_size = opt.batch_size
        self.img_size = opt.img_size
        self.save_path = opt.save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.checkpoint = opt.checkpoint
        self.examples = examples

        self.generator = Generator(in_channels=self.config.input_channels, out_channels=3, features=64).to(
            device=self.config.device)
        self.opt_gen = optim.Adam(self.generator.parameters(), lr=self.config.lr, betas=(0.5, 0.999))
        load_checkpoint(self.checkpoint, self.generator, self.opt_gen, self.config.lr, device=self.config.device)

        self.val_dataset = ArtDataset(dir_path=os.path.join(self.data_path, "test"), augment=False,
                                      img_size=self.img_size)
        self.val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=True,
                                     num_workers=self.config.num_workers)

    def inference_loop(self):
        loop = tqdm(self.val_loader, leave=True)
        self.generator.eval()
        for idx, (inputs, targets) in enumerate(loop):
            inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)
            with torch.no_grad():
                _, y_fake = self.generator(inputs)
                y_fake = y_fake * 0.5 + 0.5  # remove normalization
                torchvision.utils.save_image(y_fake, os.path.join(self.save_path, f"y_gen_{idx}.png"))
                torchvision.utils.save_image(inputs * 0.5 + 0.5, os.path.join(self.save_path, f"input_{idx}.png"))
                if idx == 1:
                    torchvision.utils.save_image(targets, os.path.join(self.save_path, f"label_{idx}.jpg"))
            if idx > self.examples:
                break


def infer():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./sample_data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--save_path", type=str, default="./samples")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/generator_300.pth")
    opt = parser.parse_args()
    inferer = Inference(opt)
    inferer.inference_loop()


if __name__ == "__main__":
    infer()
