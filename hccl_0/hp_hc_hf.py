import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# 假设你已经有了模型的预测结果和真实标签
predicted_labels = np.array([...])  # 模型的预测结果
true_labels = np.array([...])  # 真实标签

# 定义CIFAR-100的层次结构
fine_labels = ['apple', 'orange', 'pear', ...]  # 精细类别标签
coarse_labels = ['fruit', 'vehicle', 'furniture', ...]  # 粗略类别标签

# 计算精细类别的层次精度、层次召回率和层次F指标
fine_precision = precision_score(true_labels, predicted_labels, average='macro')
fine_recall = recall_score(true_labels, predicted_labels, average='macro')
fine_f1 = f1_score(true_labels, predicted_labels, average='macro')

# 计算粗略类别的层次精度、层次召回率和层次F指标
coarse_true_labels = np.array([coarse_labels[fine_labels.index(label)] for label in true_labels])
coarse_predicted_labels = np.array([coarse_labels[fine_labels.index(label)] for label in predicted_labels])

coarse_precision = precision_score(coarse_true_labels, coarse_predicted_labels, average='macro')
coarse_recall = recall_score(coarse_true_labels, coarse_predicted_labels, average='macro')
coarse_f1 = f1_score(coarse_true_labels, coarse_predicted_labels, average='macro')

print("Fine-grained Metrics:")
print(f"Precision: {fine_precision}")
print(f"Recall: {fine_recall}")
print(f"F1-score: {fine_f1}")

print("Coarse-grained Metrics:")
print(f"Precision: {coarse_precision}")
print(f"Recall: {coarse_recall}")
print(f"F1-score: {coarse_f1}")
