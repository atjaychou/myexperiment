import torch
from torch import nn, optim
from sklearn.metrics import classification_report
import yaml
import logging
import datetime
import torchvision
from simclr1.model.classifiermodel import SimCLRClassifier
from hccl.cifar100_emb_train import get_data_loaders
from hccl.sparse2coarse import sparse2coarse
from torchvision import transforms
from torch.utils.data import DataLoader

from simsiam.model.simsiam_classifier import SimSiamClassifier

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def classifier_model(checkpath, n_classes):
    freeze_base = True
    model = SimSiamClassifier(n_classes, freeze_base, checkpath)

    return model


def train_classifier(model, train_loader, test_loader, learning_rate):
    # 训练模型
    acc_list = []
    num_epoch = 50
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.007, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    # 配置日志记录器
    now = datetime.datetime.now()
    time_string = now.strftime("%m%d%H%M")
    logging.basicConfig(filename='./result/classification_report_'+time_string+'.log', level=logging.INFO)
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
        report = evaluate(test_loader, model)

        # 将报告写入日志文件
        logging.info(report)


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
def get_test_loader():
    data_dir = '../data'
    batch_size = 128
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                               std=[0.2675, 0.2565, 0.2761])])
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                              std=[0.2675, 0.2565, 0.2761])])

    train_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    dataset = config["dataset"]
    train_loader, test_loader = get_test_loader()
    checkpath = config["checkpath"]
    # 0.001-0.0001
    classify_learning_rate = config["classify_learning_rate"]
    n_classes = config["n_classes_fine"]
    model = classifier_model(checkpath, n_classes)
    # train_classifier(model, train_loader, test_loader, learning_rate)
    train_classifier(model, train_loader, test_loader, classify_learning_rate)
    # evaluate(test_loader, model)


if __name__ == '__main__':
    main()
