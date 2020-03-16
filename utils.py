import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np 
import os
import sys


def get_imagefolder_dataset(root ,**kwargs):
    dataset = ImageFolder(root, **kwargs)
    return dataset


def get_loader(dataset, **kwargs):
    return DataLoader(dataset, **kwargs)


if __name__ == "__main__":
    transforms = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor()
    ])

    dataset = get_imagefolder_dataset("data/seg_train/", transform = transforms)
    data_loader = get_loader(dataset, batch_size=3, shuffle=True)

