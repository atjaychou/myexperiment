# PyTorch相关：
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL.Image import Image
from sklearn.metrics import pairwise_distances
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Subset, DataLoader

# 数据集处理相关：
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import to_pil_image

# 聚类相关：
from sklearn.cluster import KMeans
import numpy as np

# 可视化相关：
import matplotlib.pyplot as plt

# 其他：
import torch.optim as optim
from info_nce import InfoNCE
from tqdm import tqdm
from sklearn.preprocessing import normalize

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

features_reshape = nn.Sequential(
    nn.Linear(1000, 224 * 224 * 3),
    nn.ReLU(),
).to(device)


def load_dataset(dataset_name):
    if dataset_name == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                                download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                               download=True, transform=transforms.ToTensor())
    elif dataset_name == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='../data', train=True,
                                                 download=False, transform=transforms.ToTensor())
        testset = torchvision.datasets.CIFAR100(root='../data', train=False,
                                                download=False, transform=transforms.ToTensor())
    elif dataset_name == 'mnist':
        trainset = torchvision.datasets.MNIST(root='../data', train=True,
                                              download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.MNIST(root='../data', train=False,
                                             download=True, transform=transforms.ToTensor())
    else:
        raise ValueError('Invalid dataset name')

    return trainset, testset


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

    augmented_trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform)

    augmented_train_loader = torch.utils.data.DataLoader(augmented_trainset, batch_size=32, num_workers=2,
                                                         sampler=sampler)

    # 查看数据集的大小
    # print(len(augmented_train_loader.dataset))

    train_index = indices_train

    return train_loader, test_loader, augmented_train_loader, train_index


# def get_data_loaders(dataset_name, batch_size, image_size):
#     # SubsetRandomSampler索引对应者
#     simpler_train = 500
#     indices_train = torch.randperm(len(trainset))[:simpler_train]
#     sampler = torch.utils.data.SubsetRandomSampler(indices_train)
#
#     # train_loader = torch.utils.data.DataLoader(
#     #     trainset, batch_size=batch_size, shuffle=True, num_workers=2)
#
#     # 注意，这里是为了数据方便debug，将采样方式设置为了
#     train_loader = torch.utils.data.DataLoader(
#         trainset, batch_size=batch_size, num_workers=2, sampler=sampler)
#     test_loader = torch.utils.data.DataLoader(
#         testset, batch_size=batch_size, shuffle=False, num_workers=2)
#
#     print(111)
#
#     transform = data_augmentation(image_size)
#
#     augmented_trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
#                                                       download=True, transform=transform)
#
#     # augmented_train_loader = torch.utils.data.DataLoader(
#     #     augmented_trainset, batch_size=batch_size, shuffle=False, num_workers=2)
#
#     augmented_train_loader = torch.utils.data.DataLoader(
#         augmented_trainset, batch_size=batch_size, num_workers=2, sampler=sampler)
#     return train_loader, test_loader, augmented_train_loader


def pretrained_model():
    # 加载预训练的ResNet18模型
    model_path = '../data/resnet18-5c106cde.pth'
    model = torchvision.models.resnet18(pretrained=False)
    model.load_state_dict(torch.load(model_path))
    return model


# 提取特征向量函数
def extract_features(loader, model):
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            features.append(model(images).detach())
            labels.append(targets)
    return torch.cat(features), torch.cat(labels)


# 数据变换后进行聚类？
def first_clustering(num_clusters, train_features):
    # 对特征向量进行聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    train_features_np = train_features.numpy().reshape(train_features.size(0), -1)
    cluster_labels = kmeans.fit_predict(train_features_np)
    cluster_centers = kmeans.cluster_centers_

    return cluster_labels, cluster_centers


def cluster_and_concat(features, cluster_labels):
    """
    将特征按照聚类信息聚到同一簇，并将其拼接起来
    :param features: 特征表示矩阵，维度为[num_samples, feature_dim]
    :param cluster_labels: 聚类信息，维度为[num_samples]
    :return: 返回聚类后并拼接起来的特征字典
    """
    # 初始化特征字典
    feature_dict = {}
    for i in range(len(set(cluster_labels))):
        feature_dict[i] = []

    # 按照聚类信息将特征拼接到对应的簇中
    for j in range(len(features)):
        cluster_id = cluster_labels[j]
        feature_dict[cluster_id].append(features[j])

    # 将每个簇中的特征拼接起来
    for k in feature_dict.keys():
        feature_dict[k] = torch.stack(feature_dict[k], dim=0)

    return feature_dict


def l_ij(i, j, representations, similarity_matrix, temperature, cluster_labels):
    similarity_matrix = similarity_matrix.to(device)
    temperature = temperature.to(device)
    representations = representations.to(device)
    # similarity_matrix.to(device)
    # tensor = torch.tensor([temperature], device=device)
    z_i_, z_j_ = representations[i], representations[j]
    sim_i_j = similarity_matrix[i, j]

    # 计算分子
    numerator = torch.exp(sim_i_j / temperature)

    # 计算分母中的负样本部分
    same_cluster_mask = torch.eq(cluster_labels[i], cluster_labels)
    # 获取 a 的长度
    n = len(same_cluster_mask)
    print(n) # 500
    # 创建一个全是 False 的二倍长度张量
    temp_false = torch.zeros(2 * n, dtype=torch.bool)
    temp_false[:n] = same_cluster_mask
    same_cluster_mask = temp_false.to(device)
    mask_neg = torch.ones_like(similarity_matrix).to(device) - \
               torch.matmul(same_cluster_mask.unsqueeze(0).int().float().t(),
                            same_cluster_mask.unsqueeze(0).int().float()) - \
               torch.eye(representations.shape[0]).to(device)
    mask_neg = mask_neg.clamp(min=0)

    # 计算分母中的伪正样本部分
    mask_fake_pos = torch.matmul(same_cluster_mask.unsqueeze(0).int().float().t(),
                                 same_cluster_mask.unsqueeze(0).int().float())
    # 除去样本自己
    mask_fake_pos[i] = 0

    mask_neg = mask_neg.to(device)
    mask_fake_pos = mask_fake_pos.to(device)

    # mask_neg = mask_neg.bool()
    # mask_fake_pos = mask_fake_pos.bool()

    denominator = torch.sum(
        mask_fake_pos * (torch.exp(similarity_matrix[i, :] / temperature)).to(device)
    ) + torch.sum(
        mask_neg * (torch.exp(similarity_matrix[i, :] / temperature)).to(device)
    )

    loss_ij = -torch.log(numerator / denominator)

    # 计算损失
    loss_ij = -torch.log(numerator / denominator)

    # print(loss_ij)

    return loss_ij.squeeze(0)


def my_loss(train_features, augmented_features, cluster_labels, temperature):
    cluster_labels = torch.from_numpy(cluster_labels).to(device)

    z_i = F.normalize(train_features, dim=1)
    z_j = F.normalize(augmented_features, dim=1)

    representations = torch.cat([z_i, z_j], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
    print(len(similarity_matrix))

    batch_size = z_i.size(0)

    similarity_matrix = similarity_matrix.to(device)
    temperature = torch.tensor([temperature], device=device)
    representations = representations.to(device)

    print(representations.shape[0])

    N = batch_size
    loss = 0.0
    for k in range(0, N):
        loss = 0.0
        loss += l_ij(k, k + N, representations, similarity_matrix, temperature, cluster_labels)

    return 1.0 / N * loss


def train(model, train_loader, augmented_train_loader, cluster_labels, learning_rate,
          temperature=0.5,
          margin=1.0, batch_size=64, num_epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for train_data, augmented_train_data in zip(train_loader, augmented_train_loader):
            input1, labels = train_data[0].to(device), train_data[1].to(device)
            input2, labels = augmented_train_data[0].to(device), augmented_train_data[1].to(device)

            features1 = model(input1)
            features2 = model(input2)

            optimizer.zero_grad()

            loss = my_loss(features1, features2, cluster_labels, temperature).to(device)

            loss = loss.requires_grad_()

            loss.backward()
            optimizer.step()

        print("epoch", epoch, "loss:", loss.detach().item())
        running_loss += loss.detach().item()

        # train_features, augmented_features = model(train_features, augmented_features)

    print("epoch loss:", running_loss / num_epochs)


def main():
    dataset_name = "cifar100"
    batch_size = 32
    image_size = 64

    # train_loader, train_loader1 = get_data_loaders1(dataset_name, batch_size, image_size),
    train_loader, test_loader, augmented_train_loader, train_index = get_data_loaders()

    print(111)
    # 预训练模型
    model1 = pretrained_model()

    # 提取训练集的特征向量
    train_features, train_labels = extract_features(train_loader, model1)
    augmented_features, augmented_labels = extract_features(augmented_train_loader, model1)

    model = pretrained_model()

    model = model.to(device)

    num_clusters = 100
    cluster_labels, cluster_centers = first_clustering(num_clusters, train_features)

    feature_dict = cluster_and_concat(train_features, cluster_labels)

    num_epochs = 10
    learning_rate = 0.01
    train(model, train_loader, augmented_train_loader, cluster_labels, learning_rate)

    print(222)


if __name__ == '__main__':
    main()
