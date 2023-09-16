import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import torch
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class CIFAR100KMeans:
    def __init__(self, n_clusters=100):
        # 导入 CIFAR-100 数据集
        self.cifar100_train = datasets.CIFAR100(root='./data', train=True, download=True)

        # 准备数据
        self.X = []
        self.y_true = []
        for i in range(len(self.cifar100_train)):
            img, label = self.cifar100_train[i]
            self.X.append(np.array(img).flatten())
            self.y_true.append(label)
        self.X = np.array(self.X)
        self.y_true = np.array(self.y_true)

        # 选择 k-means 算法
        self.kmeans = KMeans(n_clusters=n_clusters)

    def cluster(self):
        # 对数据进行聚类
        self.y_pred = self.kmeans.fit_predict(self.X)

    def evaluate(self):
        # 对聚类结果进行评估
        self.ari = adjusted_rand_score(self.y_true, self.y_pred)
        print(f"Adjusted Rand Index: {self.ari:.4f}")

    def visualize(self):
        # 使用 PCA 降维到二维
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X)

        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=self.y_pred, s=5)
        plt.show()
