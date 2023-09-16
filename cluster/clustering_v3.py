# PyTorch相关：
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL.Image import Image
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
import random
from tqdm import tqdm

import simCLR

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 定义transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

features_reshape = nn.Sequential(
    nn.Linear(1000, 224 * 224 * 3),
    nn.ReLU(),
)


def dataset_deal():
    # 加载CIFAR-100数据集
    train_dataset = datasets.CIFAR100(root='../data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root='../data', train=False, download=True, transform=transform)

    # 随机采样，只保留500个样本
    sinmpler_train = 500
    indices_train = torch.randperm(len(train_dataset))[:sinmpler_train]
    sampler = torch.utils.data.SubsetRandomSampler(indices_train)

    # 定义训练集、验证集和测试集的DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    # train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    train_subset = Subset(train_dataset, indices_train)

    # 查看数据集的大小
    print(len(train_loader.dataset))

    return train_loader, test_loader


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


# 定义函数将特征向量转换成图像,格式为dataloader
def convert_features_to_images(train_features, resnet):
    images = []

    for i in range(train_features.shape[0]):
        # 将train_features中的元素分别转换为能输入到resnet18中的数据
        feat = train_features[i]
        with torch.no_grad():
            img = features_reshape(feat)
            img = img.view(-1, 3, 224, 224)
            img = resnet(img)
        images.append(img)

    # 返回图像数据集
    return images


def pretraining_dataset_wrapper(ds, target_size=(96, 96), debug=False):
    randomize = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(target_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.8),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def __getitem_internal__(idx):
        this_image_raw, _ = ds[idx]
        this_image_raw = to_pil_image(this_image_raw)

        if debug:
            this_image_raw = torch.squeeze(this_image_raw)
            random.seed(idx)
            t1 = randomize(this_image_raw)
            random.seed(idx + 1)
            t2 = randomize(this_image_raw)
        else:
            t1 = randomize(this_image_raw)
            t2 = randomize(this_image_raw)

        # with torch.no_grad():
        #     t1 = t1.unsqueeze(0)
        #     t1 = model(t1)
        #     t1 = torch.flatten(t1, 1)
        #     t2 = t2.unsqueeze(0)
        #     t2 = model(t2)
        #     t2 = torch.flatten(t2, 1)

        return (t1, t2), torch.tensor(0)

    def pretraining_dataset():
        return [(lambda idx: __getitem_internal__(idx))(idx) for idx in range(len(ds))]

    return pretraining_dataset()


# class PretrainingDatasetWrapper(Dataset):
#     def __init__(self, ds: Dataset, target_size=(96, 96), debug=False):
#         super().__init__()
#         self.ds = ds
#         self.debug = debug
#         self.target_size = target_size
#         if debug:
#             print("DATASET IN DEBUG MODE")
#
#         # I will be using network pre-trained on ImageNet first, which uses this normalization.
#         # Remove this, if you're training from scratch or apply different transformations accordingly
#         self.preprocess = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])
#
#         random_resized_rotation = WrapWithRandomParams(lambda angle: ResizedRotation(angle, target_size),
#                                                        [(0.0, 360.0)])
#         self.randomize = transforms.Compose([
#             transforms.RandomResizedCrop(target_size, scale=(1 / 3, 1.0), ratio=(0.3, 2.0)),
#             transforms.RandomChoice([
#                 transforms.RandomHorizontalFlip(p=0.5),
#                 transforms.Lambda(random_rotate)
#             ]),
#             transforms.RandomApply([
#                 random_resized_rotation
#             ], p=0.33),
#             transforms.RandomApply([
#                 transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)
#             ], p=0.8),
#             transforms.RandomGrayscale(p=0.2)
#         ])
#
#     def __len__(self):
#         return len(self.ds)
#
#     def __getitem_internal__(self, idx, preprocess=True):
#         this_image_raw, _ = self.ds[idx]
#
#         if self.debug:
#             random.seed(idx)
#             t1 = self.randomize(this_image_raw)
#             random.seed(idx + 1)
#             t2 = self.randomize(this_image_raw)
#         else:
#             t1 = self.randomize(this_image_raw)
#             t2 = self.randomize(this_image_raw)
#
#         if preprocess:
#             t1 = self.preprocess(t1)
#             t2 = self.preprocess(t2)
#         else:
#             t1 = transforms.ToTensor()(t1)
#             t2 = transforms.ToTensor()(t2)
#
#         return (t1, t2), torch.tensor(0)
#
#     def __getitem__(self, idx):
#         return self.__getitem_internal__(idx, True)
#
#     def raw(self, idx):
#         return self.__getitem_internal__(idx, False)


# 数据变换后进行聚类？
def first_clustering(num_clusters, train_features):
    # 对特征向量进行聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    train_features_np = train_features.numpy().reshape(train_features.size(0), -1)
    cluster_labels = kmeans.fit_predict(train_features_np)
    cluster_centers = kmeans.cluster_centers_

    return cluster_labels, cluster_centers


# 将同一类别的特征向量用于对比学习
# def generate_pairs(num_clusters, labels):
#     pairs = []
#     for i in range(num_clusters):
#         indices = np.where(labels == i)[0]
#         for j in range(len(indices)):
#             for k in range(j + 1, len(indices)):
#                 pairs.append((indices[j], indices[k]))
#     return pairs


# 对比学习函数
# def contrastive_loss(features1, features2, margin):
#     dist = F.pairwise_distance(features1, features2)
#     # 距离的平方 +
#     loss = torch.mean((1 - margin) * dist ** 2 + margin * torch.clamp(2 - dist ** 2, min=0))
#     return loss


def contrastive_loss(emb_i, emb_j, cluster_labels, batch_size, temperature=0.5):
    z_i = F.normalize(emb_i, dim=1)
    z_j = F.normalize(emb_j, dim=1)

    sim_matrix = F.cosine_similarity(z_i.unsqueeze(1), z_j.unsqueeze(0), dim=2)

    sim_pos = torch.diagonal(sim_matrix, offset=0, dim1=0, dim2=1).reshape(-1, 1)
    sim_pos = torch.cat([sim_pos, sim_pos], dim=0)

    mask = cluster_labels.reshape(-1, 1) == cluster_labels.reshape(1, -1)
    sim_neg = sim_matrix[mask].reshape(batch_size, -1).max(dim=1).values
    sim_neg = torch.cat([sim_neg, sim_neg], dim=0)

    sim_other = sim_matrix[~mask].reshape(batch_size, -1).min(dim=1).values
    sim_other = torch.cat([sim_other, sim_other], dim=0)

    logits = torch.cat([sim_pos, sim_neg, sim_other], dim=1) / temperature
    labels = torch.zeros(batch_size * 2).to(logits.device)
    labels[:batch_size] = 1
    loss = F.cross_entropy(logits, labels.long())

    return loss


def train(pairs, train_features, train_labels, val_loader, model, num_epochs, learning_rate, margin):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()
    model = model.to(device)
    train_labels = train_labels.to(device)
    train_features = train_features.to(device)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (pair1, pair2) in enumerate(tqdm(pairs)):
            optimizer.zero_grad()
            features1 = train_features[pair1]
            features1 = features_reshape(features1)
            features1 = features1.view(-1, 3, 224, 224)
            features2 = train_features[pair2]
            features2 = features_reshape(features2)
            features2 = features2.view(-1, 3, 224, 224)

            label1 = train_labels[pair1]
            label2 = train_labels[pair2]
            if label1 == label2:
                target = torch.tensor([1])
            else:
                target = torch.tensor([0])
            features1 = model(features1)
            features2 = model(features2)
            loss = contrastive_loss(features1, features2, margin)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i + 1) % 350 == 0:
                print('Epoch [%d/%d], Batch [%d/%d], Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, len(pairs), running_loss / 350))
                running_loss = 0.0
        print('[Epoch %d] Loss: %.4f' % (epoch + 1, running_loss / len(pairs)))



def test_model(model, test_loader):
    model.eval()

    # 初始化变量
    correct = 0
    total = 0

    # 进行预测
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 计算分类准确率
    accuracy = 100 * correct / total
    print('Test Accuracy: {} %'.format(accuracy))


def main():
    # 获取数据
    train_loader, test_loader = dataset_deal()

    train_set = train_loader.dataset

    # 训练集数据进行数据变换
    train_enhanced_dataset = pretraining_dataset_wrapper(train_set, target_size=(96, 96), debug=False)


    # 预训练模型
    model = pretrained_model()

    # 提取训练集的特征向量
    train_features, train_labels = extract_features(train_loader, model)
    train_enhanced_features, train_enhanced_labels = extract_features(train_enhanced_dataset, model)


    # # 获取到数据的表征
    # images = convert_features_to_images(train_features, model)
    #
    # # 将数据拼接成数据集，500个样本
    # # concat_ds_first = TensorDataset(torch.stack(images), torch.tensor(train_labels))
    # concat_ds_first = TensorDataset(torch.stack(images), torch.tensor(train_labels))
    #
    # # 将数据集进行随机裁剪等操作后，又生成500个样本
    # enhanced_dataset_first = pretraining_dataset_wrapper(concat_ds_first, target_size=(96, 96), debug=False)

    print(111)

    enhanced_loader = DataLoader(enhanced_dataset_first, batch_size=32, shuffle=False)

    enhanced_features, enhanced_labels = extract_features(enhanced_loader, model)

    # 对原数据样本进行聚类
    # 第一次聚类
    num_clusters = 100
    cluster_labels, cluster_centers = first_clustering(num_clusters, train_features)

    image1 = torch.reshape(features_reshape(images[0]), (3, 224, 224))

    # 所有需要对比的样本对
    # pairs = generate_pairs(num_clusters, cluster_labels)
    # # print(111)

    # 调用对比学习函数进行训练
    num_epochs = 10
    learning_rate = 0.001
    margin = 0.2
    # train(pairs, train_features, train_labels, model, num_epochs, learning_rate, margin)

    # 保存模型到文件
    trained_model = 'model_100.pth'
    torch.save(model.state_dict(), trained_model)
    # ===================================
    #     第一层训练完成，测试其效果
    # ===================================
    # 加载模型
    model = torch.load(trained_model)

    # test_model(model, test_loader)


if __name__ == '__main__':
    main()
