import os
import random
import argparse
import ast

from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights, resnext50_32x4d, ResNeXt50_32X4D_Weights

from datasets import ImageDataset
from models import get_model


def generate_similarity(opt):
    """
    for every example read from the dataloader, find the best match embedding csv using cosine similarity for top k examples using opt.num_samples
    """
    headless_model = get_model(opt, backbone='resnext50_32x4d')
    dataset = ImageDataset(dir_path=opt.data_path, augment=False, img_size=opt.img_size)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=8, shuffle=False)
    df = pd.read_csv(opt.path_to_csv)
    for idx, (inputs) in enumerate(tqdm(dataloader)):
        inputs = inputs.to(opt.device)
        embedding = headless_model(inputs)
        embedding = embedding.view(-1, 2048*1*1)
        df["similarity"] = df["embedding"].apply(lambda x: torch.cosine_similarity(embedding, torch.tensor(ast.literal_eval(x)[0]).view(1, -1), dim=1).item())
        # sort by similarity
        df = df.sort_values(by=["similarity"], ascending=False)
        # plot top k images with highest similarity
        fig, axes = plt.subplots(1, opt.num_samples+1, figsize=(30, 5))
        for i in range(opt.num_samples+1):
            if i==0:
                axes[0].imshow(Image.open(os.path.join(dataset.path_to_images,dataset.list_files[idx])))
                axes[0].set_title("query")
            else:
                axes[i].imshow(plt.imread(df.iloc[i]["image_path"]))
                axes[i].set_title(f"similarity: {df.iloc[i]['similarity']: .2f}")
        plt.tight_layout()
        plt.show()
        df = df.drop(columns=["similarity"])

        if idx == 4:
            break


def generate_embeddings(opt):

    headless_model = get_model(opt, backbone='resnext50_32x4d')
    dataset = ImageDataset(dir_path=opt.data_path, augment=False, img_size=opt.img_size)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=8, shuffle=False)
    df = pd.DataFrame(columns=["image_path", "embedding"])
    for idx, (inputs) in enumerate(tqdm(dataloader)):
        inputs = inputs.to(opt.device)
        embedding = headless_model(inputs)
        embedding = embedding.view(-1, 2048*1*1)
        path_to_image = os.path.join(dataset.path_to_images, dataset.list_files[idx])
        df.loc[len(df.index)] = [path_to_image, embedding.detach().cpu().numpy().tolist()]

    os.makedirs("./results", exist_ok=True)
    df.to_csv(os.path.join("./results", "embedding.csv"))


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--run", type=str,
                             default="generate_embeddings")
    args_parser.add_argument("--data_path", type=str, default="/home/yotam/private/LootedArt/data/flowers/train")
    args_parser.add_argument("--device", type=str, default="cpu")
    args_parser.add_argument("--num_samples", type=int, default=10)
    args_parser.add_argument("--img_size", type=int, default=256)
    args_parser.add_argument("--path_to_csv", type=str, default="./results/embedding.csv")

    opt = args_parser.parse_args()

    if opt.run == "generate_embeddings":
        generate_embeddings(opt)
    elif opt.run == "generate_similarity":
        generate_similarity(opt)