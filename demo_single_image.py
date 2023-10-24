import os
import time
import datetime
import logging
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import distributed as dist
import torchvision
from torchvision import datasets, models, transforms
from configs.default_img_single import get_img_config
from models.img_resnet import ResNet50
from PIL import Image

def parse_option():
    parser = argparse.ArgumentParser(
        description='Train clothes-changing re-id model with clothes-based adversarial loss')
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    # Datasets
    parser.add_argument('--root', type=str, help="your root path to data directory")
    # Miscs
    parser.add_argument('--img_path', type=str, help='path to the image')
    parser.add_argument('--weights', type=str, help='path to the weights')
    parser.add_argument('--gpu', type=str, default='0', help='gpu id')

    args, unparsed = parser.parse_known_args()
    config = get_img_config(args)
    return config, args

@torch.no_grad()
def extract_img_feature(model, img):
    flip_img = torch.flip(img, [3])
    img, flip_img = img.cuda(), flip_img.cuda()
    _, batch_features = model(img)
    _, batch_features_flip = model(flip_img)
    batch_features += batch_features_flip
    batch_features = F.normalize(batch_features, p=2, dim=1)
    features = batch_features.cpu()

    return features

config, args = parse_option()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

dict = torch.load(args.weights)
model = ResNet50(config)
model.load_state_dict(dict['model_state_dict'])
model = model.cuda()
model.eval()

# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD = [0.229, 0.224, 0.225]
# GRID_SPACING = 10

data_transforms = transforms.Compose([
        transforms.Resize((config.DATA.HEIGHT, config.DATA.WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open(args.img_path)
image_tensor = data_transforms(image)
input_batch = image_tensor.unsqueeze(0)  # Add a batch dimension

feature = extract_img_feature(model, input_batch)

print("Input Image:", args.img_path, " Output Feautre:", feature)