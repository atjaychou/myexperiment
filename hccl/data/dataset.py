# -*- coding: utf-8 -*-
import os

from torch.utils.data import DataLoader, Dataset
import torchvision.datasets.voc as voc
import torch
from PIL import Image


class PascalVOC_Dataset(voc.VOCDetection):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

        Args:
            root (string): Root directory of the VOC Dataset.
            year (string, optional): The dataset year, supports years 2007 to 2012.
            image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
                (default: alphabetic indexing of VOC's 20 classes).
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, required): A function/transform that takes in the
                target and transforms it.
    """

    def __init__(self, root, year='2012', image_set='train', download=False, transform=None, target_transform=None):
        super().__init__(
            root,
            year=year,
            image_set=image_set,
            download=download,
            transform=transform,
            target_transform=target_transform)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
    
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        return super().__getitem__(index)

    def __len__(self):
        """
        Returns:
            size of the dataset
        """
        return len(self.images)
# 创建自定义数据集
class DogDataset(Dataset):
    def __init__(self, df, img_path, transform=None):
        self.df = df
        self.img_path = img_path
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        path = os.path.join(self.img_path, self.df.id[idx]) + '.jpg'
        img = Image.open(path)
        if self.transform:
            img = self.transform(img)

        label = self.df.label_idx[idx]
        return img, label


class DogDatasetTest(Dataset):
    def __init__(self, img_path, transform=None):
        self.img_path = img_path
        self.img_list = os.listdir(img_path)
        self.img_list.sort()
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        path = os.path.join(self.img_path, self.img_list[idx])
        img = Image.open(path)
        if self.transform:
            img = self.transform(img)

        return img
