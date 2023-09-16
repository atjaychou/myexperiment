import os
import matplotlib
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
import torch.optim as optim
import torch.multiprocessing
import logging
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy('file_system')
matplotlib.use('Agg')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

features_reshape = nn.Sequential(
    nn.Linear(1000, 224 * 224 * 3),
    nn.ReLU(),
).to(device)


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
    # 设置日志级别为WARNING
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # 定义数据集路径和是否下载的标志
    data_dir = '../data'
    download = False

    # 判断数据集文件是否已经存在
    if not os.path.exists(data_dir):
        # 如果文件不存在，则需要下载数据集
        download = True

    seed = 2245
    # 加载CIFAR-100数据集
    train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=download, transform=transforms.ToTensor())
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=download, transform=transforms.ToTensor())

    # 随机采样，只保留500个样本
    sinmpler_train = 500
    indices_train = torch.randperm(len(train_dataset))[:sinmpler_train]
    sampler = torch.utils.data.SubsetRandomSampler(indices_train)

    # 定义训练集和测试集的DataLoader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    transform = data_augmentation(target_size)

    augmented_trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=download,
                                                       transform=transform)

    torch.manual_seed(seed=seed)
    g = torch.Generator()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              num_workers=2, drop_last=True, generator=g)

    torch.manual_seed(seed=seed)
    g = torch.Generator()
    augmented_train_loader = DataLoader(augmented_trainset, batch_size=32, shuffle=True,
                                        num_workers=2, drop_last=True, generator=g)

    train_index = indices_train

    return train_loader, test_loader, augmented_train_loader, train_index


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
    # model.eval()
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


def l_ij(i, j, representations, similarity_matrix, temperature, cluster_labels, batch_size, batch_index):
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
    start_index = batch_index * batch_size
    end_index = (batch_index + 1) * batch_size
    same_cluster_mask = torch.eq(cluster_labels[i], cluster_labels[start_index:end_index])
    # 获取 a 的长度
    n = len(same_cluster_mask)
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

    # 计算损失
    loss_ij = -torch.log(numerator / denominator)

    # print(loss_ij)

    return loss_ij.squeeze(0)


def my_loss(model, train_features, augmented_features, cluster_labels, batch_index, temperature):
    cluster_labels = torch.from_numpy(cluster_labels).to(device)

    z_i = F.normalize(train_features, dim=1)
    z_j = F.normalize(augmented_features, dim=1)

    representations = torch.cat([z_i, z_j], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    batch_size = z_i.size(0)

    similarity_matrix = similarity_matrix.to(device)
    temperature = torch.tensor([temperature], device=device)
    representations = representations.to(device)

    N = batch_size
    loss = 0.0
    for k in range(0, N):
        loss = 0.0
        loss += l_ij(k, k + N, representations, similarity_matrix, temperature, cluster_labels, batch_size, batch_index)

    # 添加L2正则化项
    weight_decay = 0.1
    l2_reg = None
    for param in model.parameters():
        if l2_reg is None:
            l2_reg = param.norm(2)
        else:
            l2_reg = l2_reg + param.norm(2)

    loss = loss + weight_decay * l2_reg

    return 1.0 / N * loss


def train(model, train_loader, augmented_train_loader, cluster_labels, learning_rate,
          temperature=0.5,
          margin=1.0, batch_size=128, num_epochs=3):
    model = model.to(device)
    print("start trainning")
    for epoch in range(num_epochs):
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        running_loss = 0.0
        for batch_index, (train_data, augmented_train_data) in enumerate(
                tqdm(zip(train_loader, augmented_train_loader))):
            input1, labels = train_data[0].to(device), train_data[1].to(device)
            input2, labels = augmented_train_data[0].to(device), augmented_train_data[1].to(device)

            train_features = model(input1)
            augmented_features = model(input2)

            optimizer.zero_grad()
            loss = my_loss(model, train_features, augmented_features, cluster_labels, batch_index, temperature).to(
                device)

            loss = loss.requires_grad_()

            loss.backward()
            optimizer.step()

            running_loss += loss.detach().item()

            if (batch_index + 1) % 100 == 0:
                tqdm.write(f"Epoch: {epoch + 1}, Batch: {batch_index + 1}, Loss: {loss.item():.4f}")
        print("epoch loss:", running_loss / num_epochs)


def main():
    dataset_name = "cifar100"
    batch_size = 32
    image_size = 64

    # train_loader, train_loader1 = get_data_loaders1(dataset_name, batch_size, image_size),
    train_loader, test_loader, augmented_train_loader, train_index = get_data_loaders()

    # 预训练模型
    model1 = pretrained_model()
    # 提取训练集的特征向量
    train_features, train_labels = extract_features(train_loader, model1)
    augmented_features, augmented_labels = extract_features(augmented_train_loader, model1)

    num_clusters = 100
    cluster_labels, cluster_centers = first_clustering(num_clusters, train_features)

    feature_dict = cluster_and_concat(train_features, cluster_labels)

    num_epochs = 10
    learning_rate = 0.01
    model2 = torchvision.models.resnet18(pretrained=True)

    train(model2, train_loader, augmented_train_loader, cluster_labels, learning_rate)


if __name__ == '__main__':
    main()
