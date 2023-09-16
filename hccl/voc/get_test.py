# Create train dataloader
from hccl.voc.utils import encode_labels
# -*- coding: utf-8 -*-
import argparse
import torch

import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCDetection
from torchvision.transforms import transforms
from torchvision.models import resnet18

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset_dir = './data/VOC'


# 类别名称
class_names = ['aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
               'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
               'person',
               'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

# 粗粒度标签类别名称
coarse_class_names = ['vehicle', 'animal', 'person', 'indoor']


# 定义自定义数据集
class VOCTestDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.voc_dataset = VOCDetection(dataset_dir, year='2007', image_set='test', download=False)

    def __len__(self):
        return len(self.voc_dataset)

    def __getitem__(self, idx):
        image, target = self.voc_dataset[idx]
        image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = target['annotation']['object'][0]['name']
        label_idx = class_names.index(label)
        coarse_label_idx = self.get_coarse_label(label_idx)

        label_tensor = torch.tensor(label_idx)
        coarse_label_tensor = torch.tensor(coarse_label_idx)

        return image, label_tensor, coarse_label_tensor

    def get_coarse_label(self, label_idx):
        if label_idx <= 7:
            return 0  # vehicle
        elif label_idx < 13:
            return 1  # animal
        elif label_idx == 13:
            return 2  # person
        else:
            return 3  # indoor
