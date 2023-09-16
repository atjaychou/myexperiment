import numpy as np
import torch
import torchvision
from sklearn.cluster import AgglomerativeClustering
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


class SemiHierarchicalClustering(torch.nn.Module):
    def __init__(self, n_labels):
        super().__init__()
        self.n_labels = n_labels

        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Flatten()
        )

        self.hierarchical_clustering = AgglomerativeClustering(n_clusters=self.n_labels, linkage='ward')

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

    def convert_clusters_to_labels(self, clusters):
        labels = np.full((len(clusters),), -1)
        for cluster in np.unique(clusters):
            mask = clusters == cluster
            if sum(mask) == 1:
                labels[mask] = self.get_cluster_label(self.hierarchical_clustering.children_[mask][0])
            else:
                labels[mask] = cluster
        return labels

    def get_cluster_label(self, cluster):
        if cluster < len(self.hierarchical_clustering.children_):
            return self.get_cluster_label(self.hierarchical_clustering.children_[cluster][0])
        else:
            return cluster - len(self.hierarchical_clustering.children_)


def main():
    # 加载数据集
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载 CIFAR-100 数据集，只使用 5000 个有标签数据和 45000 个无标签数据
    labeled_dataset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform)
    labeled_indices = np.random.choice(len(labeled_dataset), size=5000, replace=False)
    labeled_dataset = torch.utils.data.Subset(labeled_dataset, labeled_indices)

    unlabeled_dataset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform)
    unlabeled_indices = np.array([i for i in range(len(unlabeled_dataset)) if i not in labeled_indices])
    unlabeled_indices = np.random.choice(unlabeled_indices, size=45000, replace=False)
    unlabeled_dataset = torch.utils.data.Subset(unlabeled_dataset, unlabeled_indices)

    model = SemiHierarchicalClustering(n_labels=100)

    # 对有标签数据进行监督学习
    model.labeled_dataloader = DataLoader(labeled_dataset, batch_size=32, shuffle=True)
    model.train_feature_extractor()

    # 对无标签数据进行半监督聚类
    labels = model.cluster(unlabeled_dataset)

    # 打印聚类结果
    print(labels)


if __name__ == '__main__':
    main()
