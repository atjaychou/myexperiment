import torch
from cluster.ImageEmb import ImageEmbedding
import torch.nn as nn
from cluster.SimClrClassifier import SimCLRClassifier
from tqdm import tqdm
from cluster.model_with_test import get_data_loaders
from torch.utils.tensorboard import SummaryWriter

# 创建 SummaryWriter 对象
writer = SummaryWriter()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def classifier_model():
    n_classes = 100
    freeze_base = True
    model = SimCLRClassifier(n_classes, freeze_base, hidden_size=512)

    return model


# learning rate = 0.0005, get best
def train_classifier(model, train_loader):
    # 训练模型
    num_epoch = 50
    learning_rate = 0.0003
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    for epoch in range(num_epoch):
        losses = 0.0
        accuracy = []
        loop = tqdm(enumerate(train_loader), total=len(train_loader), miniters=10)
        for index, (x, target) in loop:
            x = x.to(device)
            target = target.to(device)

            predict = model(x)
            loss = criterion(predict, target)
            losses += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, predictions = predict.max(1)
            num_correct = (predictions == target).sum()
            running_train_acc = float(num_correct) / float(x.shape[0])
            accuracy.append(running_train_acc)
            writer.add_scalar('Training loss', loss, global_step=epoch * len(train_loader) + index)
            writer.add_scalar('Training accuracy', running_train_acc, global_step=epoch * len(train_loader) + index)
            # writer.add_hparams({'lr':learning_rate,'batch_size':batch_size},{'acciracy':sum(accuracy)/len(accuracy)})
            # 更新信息
            loop.set_description(f'Epoch [{epoch}/{num_epoch}]')
            loop.set_postfix(loss=loss.item(), acc=running_train_acc)
        # 计算平均loss和平均准确率
        epoch_loss = losses / len(train_loader)
        epoch_acc = sum(accuracy) / len(accuracy)
        writer.add_scalar('Epoch loss', epoch_loss, global_step=epoch)
        writer.add_scalar('Epoch accuracy', epoch_acc, global_step=epoch)
        print(f"Epoch [{epoch}/{num_epoch}]: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}")
        # for i, data in enumerate(train_loader, 0):
        #     inputs, labels = data[0].to(device), data[1].to(device)
        #     optimizer.zero_grad()
        #     outputs = model(inputs)
        #     loss = criterion(outputs, labels)
        #     loss.backward()
        #     optimizer.step()
        #     running_loss += loss.item()
        #     if i % 100 == 99:
        #         print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
        #         running_loss = 0.0


# def get_acc(model, test_loader):
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in test_loader:
#             images, labels = data[0].to(device), data[1].to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#     print('Accuracy of the network on the 10000 test images: %d %%' % (
#             100 * correct / total))


def main():
    train_loader, test_loader, augmented_train_loader = get_data_loaders()
    model = classifier_model()
    train_classifier(model, train_loader)
    # get_acc(model, test_loader)


if __name__ == '__main__':
    main()
