import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from sklearn.cluster import AgglomerativeClustering

# 载入 CIFAR100 数据集
trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=torchvision.transforms.ToTensor())

# 将训练集分成带标签和不带标签两个子集
n_labels = 10
n_unlabeled = len(trainset) - n_labels
label_indices = np.concatenate([np.where(np.array(trainset.targets) == i)[0][:5] for i in range(n_labels)])
unlabeled_indices = np.array([i for i in range(len(trainset)) if i not in label_indices])
labeled_dataset = torch.utils.data.Subset(trainset, label_indices)
unlabeled_dataset = torch.utils.data.Subset(trainset, unlabeled_indices)


class SemiHierarchicalClustering:
    def __init__(self, labeled_dataset, unlabeled_dataset, n_labels):
        self.n_labels = n_labels

        # 定义用于训练卷积神经网络的数据加载器
        self.labeled_dataloader = DataLoader(labeled_dataset, batch_size=32, shuffle=True)

        # 定义用于提取特征的卷积神经网络
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 8 * 8, 128),
            torch.nn.ReLU()
        )

        # 定义用于聚类的层次聚类算法
        self.hierarchical_clustering = AgglomerativeClustering(n_clusters=n_labels)

        # 定义用于将聚类结果转换成标签的函数
        self.convert_clusters_to_labels = np.vectorize(lambda x: self.get_cluster_label(x))

        # 训练特征提取器
        self.train_feature_extractor()

    def forward(self, x):
        features = self.feature_extractor(x)
        return features

    def train_feature_extractor(self):
        feature_extractor_optimizer = torch.optim.Adam(self.feature_extractor.parameters())
        feature_extractor_criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(10):
            for images, labels in self.labeled_dataloader:
                feature_extractor_optimizer.zero_grad()

                features = self.forward(images)
                logits = torch.nn.functional.softmax(torch.nn.Linear(128, self.n_labels)(features), dim=-1)
                loss = feature_extractor_criterion(logits, labels)

                loss.backward()
                feature_extractor_optimizer.step()

    def extract_features(self, x):
        with torch.no_grad():
            features = self.feature_extractor(x)
        return features

    def cluster(self, unlabeled_dataset):
        unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=32, shuffle=False)
        features = []
        for images, _ in unlabeled_dataloader:
            features.append(self.extract_features(images))
        features = torch.cat(features, dim=0)
        clusters = self.hierarchical_clustering.fit_predict(features.cpu().numpy())
        labels = self.convert_clusters_to_labels(clusters)
        return labels

    def get_cluster_label(self, cluster):
        if cluster == -1:
            return -1
        else:
            return self.hierarchical_clustering.labels_[cluster]
