import torch
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np


def get_data_loaders(target_size=96):  # target_size=(96, 96)
    seed = 2111
    # 加载CIFAR-100数据集
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transforms.ToTensor())

    augmented_trainset = datasets.CIFAR100(root='./data', train=True, download=True,
                                           transform=transforms.ToTensor())

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    torch.manual_seed(seed=seed)
    g = torch.Generator()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              num_workers=2, drop_last=True, generator=g)

    torch.manual_seed(seed=seed)
    g = torch.Generator()
    augmented_train_loader = DataLoader(augmented_trainset, batch_size=32, shuffle=True,
                                        num_workers=2, drop_last=True, generator=g)

    return train_loader, test_loader, augmented_train_loader


if __name__ == '__main__':
    train_loader, test_loader, augmented_train_loader = get_data_loaders()
    data1, label1 = next(iter(train_loader))
    data2, label2 = next(iter(augmented_train_loader))
    first_data1 = data1[0]
    first_data2 = data2[0]

    print('第一条数据的形状：', first_data1.shape)
    print('第一条数据的形状：', first_data2.shape)
    print('第1条数据的标签：', label1[0])
    print('第2条数据的标签：', label2[0])
