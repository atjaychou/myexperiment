import torch
from torch import nn, optim
from sklearn.metrics import classification_report
import yaml
import logging
import datetime
import torchvision
from hccl.model.classifiermodel import SimCLRClassifier
from hccl.sparse2coarse import sparse2coarse
import numpy as np
import random
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def data_augmentation(image_size):
    return transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.RandomResizedCrop(size=image_size, scale=(0.8, 1.0)),
        # transforms.RandomCrop(size=image_size, padding=int(image_size / 8)),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                             std=[0.2675, 0.2565, 0.2761])
    ])

def get_data_loaders(dataset_name, target_size=96):
    num_clusters = 0
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                               std=[0.2675, 0.2565, 0.2761])])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                              std=[0.2675, 0.2565, 0.2761])])
    # set log level as WARNING
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    data_dir = '../data'
    # download = False
    #
    # if not os.path.exists(data_dir):
    #     download = True

    seed = 2235

    if dataset_name == 'cifar10':
        # CIFAR-10 dataset
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=data_augmentation)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

        batch_size = 128

        # DataLoader
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        transform = data_augmentation(target_size)

        augmented_trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True,
                                                          transform=data_augmentation)

        torch.manual_seed(seed=seed)
        g = torch.Generator()
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=0, drop_last=True, generator=g)

        torch.manual_seed(seed=seed)
        g = torch.Generator()
        augmented_train_loader = DataLoader(augmented_trainset, batch_size=batch_size, shuffle=False,
                                            num_workers=0, drop_last=True, generator=g)
        print("File ready")

        return train_loader, test_loader, augmented_train_loader, train_dataset

    elif dataset_name == 'cifar100':
        # CIFAR-100 dataset
        train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)

        batch_size = 128

        # DataLoader
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        transform = data_augmentation(target_size)

        augmented_trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True,
                                                           transform=transform)

        torch.manual_seed(seed=seed)
        g = torch.Generator()
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=0, drop_last=True, generator=g)

        g = torch.Generator()
        augmented_train_loader = DataLoader(augmented_trainset, batch_size=batch_size, shuffle=False,
                                            num_workers=0, drop_last=True, generator=g)
        print("File ready")

        augmented_loader_1, augmented_loader_2 = train_loader, augmented_train_loader
        data_iter1 = iter(augmented_loader_1)
        images1, labels1 = next(data_iter1)
        print(labels1)
        data_iter2 = iter(augmented_loader_2)
        images2, labels2 = next(data_iter2)
        print(labels2)

        return train_loader, test_loader, augmented_train_loader, train_dataset

def classifier_model(checkpath, n_classes):
    freeze_base = True
    model = SimCLRClassifier(n_classes, freeze_base, checkpath)

    return model


def train_classifier(model, train_loader, test_loader, learning_rate):
    # 训练模型
    acc_list = []
    num_epoch = 100
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.007, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    # 配置日志记录器
    now = datetime.datetime.now()
    time_string = now.strftime("%m%d%H%M")
    logging.basicConfig(filename='./result/classification_fine_'+time_string+'.log', level=logging.INFO)
    for epoch in range(num_epoch):
        losses = 0.0
        for index, (x, target) in enumerate(train_loader):
            x = x.to(device)
            target = target.to(device)

            predict = model(x)
            loss = criterion(predict, target)
            losses += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if index % 30 == 29:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, index + 1, losses / 30))
                losses = 0.0
        # test_model(model, test_loader)
        report = evaluate(test_loader, model)
        #
        # # 将报告写入日志文件
        logging.info(report)

def test_model(model, test_loader):
    model.eval()
    # 初始化变量+
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
    # 计算分类准确率
    accuracy = 100 * correct / total
    print('Test Accuracy: {} %'.format(accuracy))


def evaluate(data_loader, module):
    with torch.no_grad():
        progress = ["/", "-", "\\", "|", "/", "-", "\\", "|"]
        module.eval().cuda()
        true_y, pred_y = [], []
        for i, batch_ in enumerate(data_loader):
            X, y = batch_
            print(progress[i % len(progress)], end="\r")
            y_pred = torch.argmax(module(X.cuda()), dim=1)
            true_y.extend(y.cpu())
            pred_y.extend(y_pred.cpu())
        report = classification_report(true_y, pred_y, digits=3)
        print(report)
        return report

def test_model_fine(PATH):
    # 设置随机数种子
    setup_seed(100)
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    dataset = config["dataset"]
    train_loader, test_loader, augmented_train_loader, _ = get_data_loaders(dataset)
    checkpath = PATH
    # 0.001-0.0001
    classify_learning_rate = config["classify_learning_rate"]
    n_classes = config["n_classes_fine"]
    model = classifier_model(checkpath, n_classes)
    # train_classifier(model, train_loader, test_loader, learning_rate)
    train_classifier(model, train_loader, test_loader, classify_learning_rate)
    # evaluate(test_loader, model)

