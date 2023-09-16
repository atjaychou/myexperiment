import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

def get_features():
    # 加载预训练的ResNet-18模型
    model = torchvision.models.resnet18(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])  # 移除最后一层全连接层
    model.eval()

    # 加载CIFAR-10数据集
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    cifar10_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, transform=transform, download=True)
    cifar10_dataloader = torch.utils.data.DataLoader(cifar10_dataset, batch_size=128, shuffle=False)

    # 提取特征向量
    features = []
    with torch.no_grad():
        for images, _ in cifar10_dataloader:
            outputs = model(images)
            features.extend(outputs.squeeze().cpu().numpy())

    features = np.array(features)
    true_labels = cifar10_dataset.targets
    return features, true_labels


def compute_ari(feature, true_labels, k, last_cluster_labels):
    current_pred_labels = []
    kmeans = KMeans(n_clusters=k, random_state=0)
    pred_labels = kmeans.fit_predict(feature)
    if len(pred_labels) == len(true_labels):
        ari = adjusted_rand_score(true_labels, pred_labels)
        centers = kmeans.cluster_centers_
        return ari, centers, pred_labels
    else:
        for i in range(len(true_labels)):
            current_pred_labels.append(pred_labels[last_cluster_labels[i]])
        ari = adjusted_rand_score(true_labels, current_pred_labels)
        centers = kmeans.cluster_centers_
        return ari, centers, current_pred_labels



def get_best_cluster(min_k, max_k, feature, true_labels, last_cluster_labels):
    best_centers = None
    best_ari = -1
    best_k = -1
    best_labels = []

    for k in range(min_k, max_k + 1):
        ari, centers, pred_labels = compute_ari(feature, true_labels, k, last_cluster_labels)
        if ari > best_ari:
            best_ari = ari
            best_k = k
            best_centers = centers
            best_labels = pred_labels
    return best_ari, best_k, best_centers, best_labels

def get_labels(pred_labels1, pred_labels2):
    new_labels = []
    for label1 in pred_labels1:
        new_label = [label1]  # 初始化新的伪标签，加入第一次聚类的标签

        # 查找该样本在第一次聚类中的标签对应的第二次聚类的标签
        label2 = pred_labels2[label1]
        new_label.append(label2)  # 将第二次聚类的标签加入新的伪标签

        new_labels.append(new_label)  # 将新的伪标签加入到新的标签列表中
    return new_labels

def get_all_labels():
    features, true_labels = get_features()
    min_k1 = 10
    max_k1 = 15
    best_ari1, best_k1, best_centers1, pred_labels1 = get_best_cluster(min_k1, max_k1, features, true_labels, last_cluster_labels=None)

    print("Best number of clusters:", best_k1)
    print("Best Adjusted Rand Index:", best_ari1)

    min_k2 = 3  # 聚类数目的最小值
    max_k2 = 5  # 聚类数目的最大值
    best_ari2, best_k2, best_centers2, pred_labels2 = get_best_cluster(min_k2, max_k2, best_centers1, true_labels, pred_labels1)

    print("Best number of clusters:", best_k2)
    print("Best Adjusted Rand Index:", best_ari2)

    hc_labels = get_labels(pred_labels1, pred_labels2)
    np.savetxt('./data/cluster_labels_ cifar10.txt', hc_labels, fmt='%d')

if __name__ == '__main__':
    get_all_labels()



