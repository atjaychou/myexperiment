import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from collections import defaultdict

# 批次大小、学习率、训练轮数等：
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 20

# 下载并准备数据集,定义一些数据增强的操作：
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

# 使用torchvision.datasets.CIFAR100类来加载数据集：
train_dataset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=False, transform=train_transform)

test_dataset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=False, transform=test_transform)


# 为了方便起见，我们可以将每个标签表示为一个包含其所有祖先标签的列表。为此，我们需要定义一个函数来将标签映射到其所有祖先标签：
def get_ancestor_labels(labels, label2ancestors):
    ancestor_labels = set()
    for label in labels:
        ancestor_labels |= set(label2ancestors[label])
    return list(ancestor_labels)

print()
# 接下来，我们可以使用这个函数将数据集中的每个标签转换为其所有祖先标签：
label2ancestors = defaultdict(list)
with open('cifar100_hierarchy.txt') as f:
    for line in f:
        child, ancestors = line.strip().split(':')
        ancestors = [int(a) for a in ancestors.split()]
        label2ancestors[int(child)] = ancestors

train_labels = []
test_labels = []

for _, label in train_dataset:
    train_labels.append(get_ancestor_labels([label], label2ancestors))

for _, label in test_dataset:
    test_labels.append(get_ancestor_labels([label], label2ancestors))


# 然后，我们可以定义一个数据加载器来加载数据集。在这个实现中，我们将每个标签表示为一个one-hot向量，并将标签和输入数据一起打包：
class CIFAR100Dataset(torch.utils.data.Dataset):
    def init(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        return img, torch.Tensor(label)

    def __len__(self):
        return len(self.data)


train_loader = DataLoader(
    CIFAR100Dataset(train_dataset.data, train_labels),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
)

test_loader = DataLoader(
    CIFAR100Dataset(test_dataset.data, test_labels),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
)


# 现在我们可以定义ResNet18作为特征提取器。我们可以使用预训练的ResNet18模型并去掉最后一层，以便获取最后一个全连接层之前的特征表示：
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        return self.resnet(x)


class LabelGenerator(nn.Module):
    def __init__(self, num_labels):
        super(LabelGenerator, self).__init__()
        self.fc = nn.Linear(512, num_labels)

    def forward(self, x):
        return self.fc(x)


# 接下来，我们可以定义标签生成器。在这个实现中，我们使用一个全连接层来生成标签，
# 并使用对比学习来学习标签之间的相似度。具体来说，我们首先将所有标签表示为one-hot向量，
# 并将其传递给标签生成器。然后，我们计算所有标签的嵌入向量，并将其归一化为单位长度。
# 接下来，我们计算所有标签之间的余弦相似度，并将其保存在一个矩阵中。
# 在每个训练步骤中，我们随机选择一个标签，并选择其最相似的标签作为下一个生成的标签：
def train_contrastive(model, optimizer, criterion, num_epochs, train_loader):
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.cuda(), labels.cuda()

            optimizer.zero_grad()

            features = model(data)
            logits = model.fc(features)

            # Calculate similarity matrix
            similarities = torch.matmul(logits, logits.t())
            similarities = similarities / torch.norm(logits, dim=1)
            similarities = similarities / torch.norm(logits.t(), dim=0)

            # Mask out current label
            batch_size = logits.size(0)
            indices = torch.arange(batch_size, device=logits.device)
            similarities[indices, indices] = float('-inf')

            # Find most similar label
            top_similarities, top_indices = similarities.topk(k=1, dim=1)
            top_labels = labels[top_indices.view(-1)]

            loss = criterion(logits, top_labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('Epoch: [{}/{}]\tBatch: [{}/{}]\tLoss: {:.6f}'.format(
                    epoch + 1, num_epochs, batch_idx + 1, len(train_loader), loss.item()))


# 定义一个测试函数，用于评估模型在测试集上的性能：
def test(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            features = model(data)
            output = model.fc(features)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    acc = 100. * correct / len(test_loader.dataset)
    print('Test set accuracy: {:.2f}%'.format(acc))


# 训练模型并评估其性能：
feature_extractor = FeatureExtractor().cuda()
label_generator = LabelGenerator(100).cuda()

optimizer = optim.Adam(list(feature_extractor.parameters()) + list(label_generator.parameters()), lr=LR)
criterion = nn.CrossEntropyLoss()

train_contrastive(feature_extractor, optimizer, criterion, NUM_EPOCHS, train_loader)
test(feature_extractor, test_loader)
