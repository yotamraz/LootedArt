import os
import random
import argparse

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights, resnext50_32x4d, ResNeXt50_32X4D_Weights

from datasets import FlowersDataset


def action(opt):
    # full_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    full_model = resnext50_32x4d(ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
    # register forward hook to get embedding
    headless_model = nn.Sequential(*list(full_model.children())[:-1]).to(opt.device)
    headless_model.eval()

    dataset = FlowersDataset(dir_path=opt.data_path, augment=False, img_size=256)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=8, shuffle=False)
    df = pd.DataFrame(columns=["image_path", "embedding"])
    for idx, (inputs) in enumerate(tqdm(dataloader)):
        inputs = inputs.to(opt.device)
        embedding = headless_model(inputs)
        embedding = embedding.view(-1, 2048*1*1)
        path_to_image = os.path.join(dataset.path_to_images, dataset.list_files[idx])
        df.loc[len(df.index)] = [path_to_image, embedding.detach().cpu().numpy().tolist()]

    df.to_csv(os.path.join("./results", "embedding.csv"))


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--data_path", type=str, default="/home/yotam/Projects/LootedArt/ResNetSim/flowers/flowers")
    args_parser.add_argument("--device", type=str, default="cuda:0")

    opt = args_parser.parse_args()
    action(opt)