import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import yaml

from hccl.cifar100_emb_train import get_data_loaders
from hccl.imageNet_dog_emb_train import get_dogdataset_loader


def get_features():
    # 加载预训练的ResNet-18模型
    model = torchvision.models.resnet18(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])  # 移除最后一层全连接层
    model.eval()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    dataset = config["dataset"]
    # train_dataloader, test_loader, augmented_train_loader, cifar100_dataset = get_data_loaders(dataset)
    train_dataloader, augmented_train_loader, test_loader = get_dogdataset_loader()

    # 提取特征向量
    features = []
    with torch.no_grad():
        for images, _ in train_dataloader:
            outputs = model(images)
            features.extend(outputs.squeeze().cpu().numpy())

    features = np.array(features)
    return features

def get_best_cluster(feature, k, last_cluster_labels):
    centers, pred_labels = compute_ari(feature, k, last_cluster_labels)
    return centers, pred_labels

def compute_ari(feature, k, last_cluster_labels):
    current_pred_labels = []
    kmeans = KMeans(n_clusters=k, random_state=0)
    pred_labels = kmeans.fit_predict(feature)   # 第一次50000个，1000类                 第二次50000个,500类
    if last_cluster_labels is None:  #   feature.shape[0]=50000                       feature.shape[0]=1000
        centers = kmeans.cluster_centers_
        return centers, pred_labels
    else:
        for i in range(feature.shape[0]):
            current_pred_labels.append(pred_labels[last_cluster_labels[i]])
        centers = kmeans.cluster_centers_
        return centers, current_pred_labels


def get_labels(pred_labels1, pred_labels2, pred_labels3):
    new_labels = []
    for label1 in pred_labels1:
        new_label = [label1]  # 初始化新的伪标签，加入第一次聚类的标签

        # 查找该样本在第一次聚类中的标签对应的第二次聚类的标签
        label2 = pred_labels2[label1]
        new_label.append(label2)  # 将第二次聚类的标签加入新的伪标签

        new_labels.append(new_label)  # 将新的伪标签加入到新的标签列表中

        label3 = pred_labels3[label2]
        new_label.append(label3)  # 将第二次聚类的标签加入新的伪标签

        new_labels.append(new_label)  # 将新的伪标签加入到新的标签列表中


    return new_labels

def get_all_labels():
    features= get_features()
    k1 = 1000
    k2 = 500
    k3 = 100
    best_centers1, pred_labels1 = get_best_cluster(features, k1, last_cluster_labels=None)

    best_centers2, pred_labels2 = get_best_cluster(best_centers1, k2, pred_labels1)

    best_centers3, pred_labels3 = get_best_cluster(best_centers2, k3, pred_labels2)

    hc_labels = get_labels(pred_labels1, pred_labels2, pred_labels3)
    np.savetxt('./data/cluster_for_train_labels_dog_1500.txt', hc_labels, fmt='%d')

if __name__ == '__main__':
    get_all_labels()



