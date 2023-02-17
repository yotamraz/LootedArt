
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, resnext50_32x4d, ResNeXt50_32X4D_Weights


def get_model(opt, backbone='resnext50_32x4d'):
    # full_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    full_model = resnext50_32x4d(ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
    # register forward hook to get embedding
    headless_model = nn.Sequential(*list(full_model.children())[:-1]).to(opt.device)
    headless_model.eval()

    return headless_model