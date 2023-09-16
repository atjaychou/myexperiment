import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import yaml

from hccl.voc.voc_train import get_data_loaders


def get_features():
    # 加载预训练的ResNet-18模型
    model = torchvision.models.resnet18(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])  # 移除最后一层全连接层
    model.eval()

    with open("../config.yaml", "r") as f:
        config = yaml.safe_load(f)
    voc_dataloader, test_loader, augmented_train_loader, voc_dataset = get_data_loaders(dataset_dir = '../data/VOC')

    # 提取特征向量
    features = []
    with torch.no_grad():
        for images, _, _ in augmented_train_loader:
            outputs = model(images)
            features.extend(outputs.squeeze().cpu().numpy())

    features = np.array(features)
    return features


def compute_ari(feature, k, last_cluster_labels):
    current_pred_labels = []
    kmeans = KMeans(n_clusters=k, random_state=0)
    pred_labels = kmeans.fit_predict(feature)
    if len(pred_labels) == feature.shape[0]:
        centers = kmeans.cluster_centers_
        return centers, pred_labels
    else:
        for i in range(feature.shape[0]):
            current_pred_labels.append(pred_labels[last_cluster_labels[i]])
        centers = kmeans.cluster_centers_
        return centers, current_pred_labels



def get_best_cluster(feature, k, last_cluster_labels):
    centers, pred_labels = compute_ari(feature, k, last_cluster_labels)
    return centers, pred_labels

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
    features= get_features()
    k1 = 30
    k2 = 5
    best_centers1, pred_labels1 = get_best_cluster(features, k1, last_cluster_labels=None)

    best_centers2, pred_labels2 = get_best_cluster(best_centers1, k2, pred_labels1)

    hc_labels = get_labels(pred_labels1, pred_labels2)
    np.savetxt('./data/cluster_labels_voc_aug_505.txt', hc_labels, fmt='%d')

if __name__ == '__main__':
    get_all_labels()



