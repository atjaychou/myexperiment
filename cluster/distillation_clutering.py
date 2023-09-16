# import torch
# import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
# import torch.nn.functional as F
# import torch
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
#
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
#
#
#
#
#
#
#
# # 定义transform
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
#
# # 加载cifar100数据集
# cifar100_train = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
#
# # 随机采样，只保留500个样本
# indices = torch.randperm(len(cifar100_train))[:2000]
# sampler = torch.utils.data.SubsetRandomSampler(indices)
#
# # 使用DataLoader加载数据
# batch_size = 32
# trainloader = torch.utils.data.DataLoader(cifar100_train, batch_size=batch_size, sampler=sampler)
#
# # 查看数据集的大小
# print(len(trainloader.dataset))
#
# # 加载预训练的ResNet18模型
# model = torchvision.models.resnet18(pretrained=True)
#
#
# # 提取特征向量函数
# def extract_features(loader, model):
#     features = []
#     labels = []
#     model.eval()
#     with torch.no_grad():
#         for images, targets in loader:
#             features.append(model(images).detach())
#             labels.append(targets)
#     return torch.cat(features), torch.cat(labels)
#
#
# # 提取训练集的特征向量
# train_features, train_labels = extract_features(trainloader, model)
# from sklearn.cluster import KMeans
# import numpy as np
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# import torch
# import torch.nn.functional as F
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
#
# # 假设已经从数据集中提取了特征向量features和对应的标签labels，以及训练好的神经网络模型teacher_model
#
# # 使用训练好的神经网络模型对特征向量进行蒸馏
# def distillation(features, teacher_model, temperature):
#     with torch.no_grad():
#         teacher_model.eval()
#         soft_targets = F.softmax(teacher_model(features/temperature), dim=1)
#     return soft_targets
#
# # 初始化聚类模型
# kmeans = KMeans(n_clusters=10, random_state=0)
#
# # 使用蒸馏学习将训练好的神经网络模型teacher_model的知识转移到聚类模型中
# temperature = 5
# soft_targets = distillation(features, teacher_model, temperature)
#
# # 将蒸馏得到的软标签作为聚类模型的训练数据，进行聚类模型的训练
# kmeans.fit(soft_targets)
#
# # 使用训练好的聚类模型对特征向量进行聚类，得到聚类标签
# predicted_labels = kmeans.predict(soft_targets)
#
# # 使用sklearn.metrics中的聚类性能指标来评估聚类质量
# silhouette_score = silhouette_score(features, predicted_labels)
# print("Silhouette score: ", silhouette_score)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# # 对特征向量进行聚类
# num_clusters = 100
# kmeans = KMeans(n_clusters=num_clusters, random_state=0)
# train_features_np = train_features.numpy().reshape(train_features.size(0), -1)
# cluster_labels = kmeans.fit_predict(train_features_np)
#
#
#
#
# # 将同一类别的特征向量用于对比学习
# def generate_pairs(labels):
#     pairs = []
#     for i in range(num_clusters):
#         indices = np.where(labels == i)[0]
#         for j in range(len(indices)):
#             for k in range(j + 1, len(indices)):
#                 pairs.append((indices[j], indices[k]))
#     return pairs
#
#
# pairs = generate_pairs(cluster_labels)
#
#
# # 对比学习函数
# def contrastive_loss(features1, features2, margin):
#     dist = F.pairwise_distance(features1, features2)
#     loss = torch.mean((1 - margin) * dist ** 2 + margin * torch.clamp(2 - dist ** 2, min=0))
#     return loss
#
#
# features_reshape = nn.Sequential(
#     nn.Linear(1000, 224 * 224 * 3),
#     nn.ReLU(),
# ).to(device)
#
#
#
# def train(pairs, train_features, train_labels, model, num_epochs, learning_rate, margin):
#     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
#     criterion = nn.CrossEntropyLoss()
#     model.train()
#     model = model.to(device)
#     train_labels = train_labels.to(device)
#     train_features = train_features.to(device)
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         for i, (pair1, pair2) in enumerate(pairs):
#             optimizer.zero_grad()
#             features1 = train_features[pair1]
#             features1 = features_reshape(features1)
#             features1 = features1.view(-1, 3, 224, 224)
#             features2 = train_features[pair2]
#             features2 = features_reshape(features2)
#             features2 = features2.view(-1, 3, 224, 224)
#
#             label1 = train_labels[pair1]
#             label2 = train_labels[pair2]
#             if label1 == label2:
#                 target = torch.tensor([1])
#             else:
#                 target = torch.tensor([0])
#             features1 = model(features1)
#             features2 = model(features2)
#             loss = contrastive_loss(features1, features2, margin)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#             if (i + 1) % 100 == 0:
#                 print('Epoch [%d/%d], Batch [%d/%d], Loss: %.4f'
#                       % (epoch + 1, num_epochs, i + 1, len(pairs), running_loss / 100))
#                 running_loss = 0.0
#         print('[Epoch %d] Loss: %.4f' % (epoch + 1, running_loss / len(pairs)))
#
#
# # 调用对比学习函数进行训练
# num_epochs = 10
# learning_rate = 0.001
# margin = 0.2
# train(pairs, train_features, train_labels, model, num_epochs, learning_rate, margin)
