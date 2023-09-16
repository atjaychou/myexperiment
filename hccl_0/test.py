import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_augmented_data_loaders(batch_size):
    # 数据增强的变换操作
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomCrop(32, padding=4),  # 随机裁剪
        transforms.ToTensor(),  # 转为Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
    ])

    # 加载训练集数据
    train_dataset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # 第一次数据增强
    augmented_dataset_1 = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform)
    augmented_loader_1 = DataLoader(augmented_dataset_1, batch_size=batch_size, shuffle=True, num_workers=2)

    # 第二次数据增强
    augmented_dataset_2 = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform)
    augmented_loader_2 = DataLoader(augmented_dataset_2, batch_size=batch_size, shuffle=True, num_workers=2)

    return augmented_loader_1, augmented_loader_2

# 使用示例

