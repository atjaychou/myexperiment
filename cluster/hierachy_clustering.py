import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.semi_supervised import LabelPropagation
import os
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import random
import torch
from sklearn.metrics.pairwise import pairwise_distances

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# 加载数据集，data是数据，labels是标签
def load_data():
    # 加载数据集并进行预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    return trainloader, testloader


def get_model():
    model = torchvision.models.resnet18(pretrained=True)
    # remove last classifier layer
    model.fc = torch.nn.Identity()
    model.to(device)
    model.eval()
    return model


# get features
def extract_features(dataloader, model):
    features = []
    labels = []
    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.to(device)
        features.append(model(data).detach().cpu())
        labels.append(target)
    print('extract_features done')

    return torch.cat(features), torch.cat(labels)


def split_data(features, labels):
    # 假设已经获取到了特征features和标签labels
    # features: shape=[num_samples, feature_dim]
    # labels: shape=[num_samples]

    num_labeled = 500  # 有标签样本数量
    num_total = features.shape[0]  # 总样本数量

    # 随机选取一部分样本作为有标签样本
    labeled_indices = random.sample(range(num_total), num_labeled)
    unlabeled_indices = list(set(range(num_total)) - set(labeled_indices))

    labeled_features = features[labeled_indices]
    labeled_labels = labels[labeled_indices]

    unlabeled_features = features[unlabeled_indices]
    unlabeled_labels = torch.zeros(len(unlabeled_indices), dtype=torch.long)  # 无标签样本的标签设置为0

    print("有标签样本数量：", len(labeled_indices))
    print("无标签样本数量：", len(unlabeled_indices))

    return labeled_features, unlabeled_features, labeled_labels





def main():
    trainloader, testloader = load_data()
    model = get_model()

    if os.path.exists('./data/train_features.pt'):
        # 文件存在的情况下执行的代码
        print("features loaded")
        train_features = torch.load('./data/train_features.pt')
        train_labels = torch.load('./data/train_labels.pt')
    else:
        train_features, train_labels = extract_features(trainloader, model)
        # test_features, test_labels = extract_features(testloader, model)

        # 保存特征向量
        torch.save(train_features, './data/train_features.pt')
        torch.save(train_labels, './data/train_labels.pt')

    labeled_features, unlabeled_features, labeled_labels = split_data(train_features, train_labels)

    # 训练一个标签传播模型
    lp = LabelPropagation(kernel='knn', n_neighbors=5)
    lp.fit(labeled_features, labeled_labels)

    # 利用标签传播模型的结果，对无标记的数据进行聚类
    unlabeled_labels = lp.transduction_[len(labeled_labels):]
    clustering = AgglomerativeClustering(n_clusters=None, linkage='ward', compute_full_tree=True)
    clustering.fit(unlabeled_features)

    # 将有标记的数据和无标记的数据的聚类结果合并
    all_labels = np.concatenate((labeled_labels, unlabeled_labels))

    # 计算所有数据的距离矩阵
    distances = pairwise_distances(train_features)

    # 对距离矩阵进行层次聚类
    final_clustering = AgglomerativeClustering(n_clusters=None, linkage='ward', affinity='precomputed')
    final_clustering.fit(distances)

    # 输出聚类结果
    print(final_clustering.labels_)


if __name__ == '__main__':
    main()
