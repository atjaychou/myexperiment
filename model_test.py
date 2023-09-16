import torch
from torch.utils.data import DataLoader, SequentialSampler
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision import transforms

# 定义transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 加载模型
model = models.resnet18().to(device)
model.load_state_dict(torch.load('model.pth'))


class CIFAR100Coarse(datasets.CIFAR100):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 构建标签的细粒度到粗粒度映射
        fine_labels = self.targets
        coarse_labels = [self.class_to_coarse[label] for label in fine_labels]
        self.coarse_labels = coarse_labels

    def __getitem__(self, index):
        img, fine_label = super().__getitem__(index)
        coarse_label = self.coarse_labels[index]
        return img, coarse_label

# test_dataset = CIFAR100Coarse(root='./data', train=False, download=True, transform=transform, class_to_coarse=class_to_coarse)


test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
print(111)


def test_model(model, test_dataset):
    # 随机选择100个样本进行测试
    sampler = SequentialSampler(range(100))
    test_loader = DataLoader(test_dataset, batch_size=32, sampler=sampler)

    model.eval()

    # 初始化变量
    correct = 0
    total = 0

    count = 0
    # 进行预测
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            count += 1
            print(count)

    # 计算分类准确率
    accuracy = 100 * correct / total
    print('Test Accuracy: {} %'.format(accuracy))


test_model(model, test_dataset)
