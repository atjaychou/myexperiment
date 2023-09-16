import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

# 定义CIFAR-100数据集的transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 加载CIFAR-100数据集
cifar100_train = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

# 获取数据集中的标签和类别名
labels = cifar100_train.targets
class_names = cifar100_train.classes

# 统计每个类别的样本数量
class_count = {class_name: labels.count(class_idx) for class_idx, class_name in enumerate(class_names)}

# 可视化类别数量
plt.figure(figsize=(12, 5))
plt.bar(class_count.keys(), class_count.values())
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.title('CIFAR-100 Class Distribution')
plt.xticks(rotation=90)
plt.show()
