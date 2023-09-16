from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Subset, DataLoader
import torchvision.datasets as datasets
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def data_augmentation(image_size):
    return transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.RandomResizedCrop(size=image_size, scale=(0.8, 1.0)),
        # transforms.RandomCrop(size=image_size, padding=int(image_size / 8)),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def get_data_loaders(target_size=96):  # target_size=(96, 96)
    # 加载CIFAR-100数据集
    train_dataset = datasets.CIFAR100(root='../data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.CIFAR100(root='../data', train=False, download=True, transform=transforms.ToTensor())

    # 随机采样，只保留500个样本
    sinmpler_train = 500
    indices_train = torch.randperm(len(train_dataset))[:sinmpler_train]
    sampler = torch.utils.data.SubsetRandomSampler(indices_train)

    # 定义训练集和测试集的DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    # train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(111)

    transform = data_augmentation(target_size)

    augmented_trainset = datasets.CIFAR100(root='../data', train=True, download=True, transform=transform)

    augmented_train_loader = torch.utils.data.DataLoader(augmented_trainset, batch_size=32, num_workers=2,
                                                         sampler=sampler)

    # 查看数据集的大小
    # print(len(augmented_train_loader.dataset))

    train_index = indices_train

    return train_loader, test_loader, augmented_train_loader, train_index


def pretrained_model():
    # 加载预训练的ResNet18模型
    model_path = '../data/resnet18-5c106cde.pth'
    model = models.resnet18(pretrained=False)
    model.load_state_dict(torch.load(model_path))
    # model = torch.load("../data/resnet18-5c106cde.pth")
    return model


if __name__ == '__main__':
    train_loader, test_loader, augmented_train_loader, train_index = get_data_loaders()
    model = pretrained_model()
    model.train()
    lr = 0.01
