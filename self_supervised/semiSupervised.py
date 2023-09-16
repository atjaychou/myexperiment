import os
import matplotlib
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
import torch.multiprocessing
import logging
import datetime
from torch.optim import RMSprop
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from self_supervised.ImageEmb import ImageEmbeddingClass
from torch.utils.tensorboard import SummaryWriter

# 创建 SummaryWriter 对象
writer = SummaryWriter()

torch.multiprocessing.set_sharing_strategy('file_system')
matplotlib.use('Agg')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])])


def get_data_loaders(target_size=96):
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
    download = False

    if not os.path.exists(data_dir):
        download = True

    seed = 2245
    # CIFAR-100 dataset
    train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=download, transform=train_transform)
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=download, transform=test_transform)

    batch_size = 128

    # DataLoader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    transform = data_augmentation(target_size)

    augmented_trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=download,
                                                       transform=transform)

    torch.manual_seed(seed=seed)
    g = torch.Generator()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, drop_last=True, generator=g)

    torch.manual_seed(seed=seed)
    g = torch.Generator()
    augmented_train_loader = DataLoader(augmented_trainset, batch_size=batch_size, shuffle=True,
                                        num_workers=2, drop_last=True, generator=g)
    print("File ready")

    return train_loader, test_loader, augmented_train_loader


# get features
def extract_features(loader, model):
    features = []
    labels = []
    # model.eval()
    with torch.no_grad():
        for images, targets in loader:
            features.append(model(images).detach())
            labels.append(targets)
    print('extract_features done')
    return torch.cat(features), torch.cat(labels)


def first_clustering(num_clusters, train_features):
    # clustering for features
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    train_features_np = train_features.numpy().reshape(train_features.size(0), -1)
    cluster_labels = kmeans.fit_predict(train_features_np)
    cluster_centers = kmeans.cluster_centers_
    print('clustering done')
    return cluster_labels, cluster_centers


def l_ij(i, j, representations, similarity_matrix, temperature, cluster_labels, batch_size, batch_index):
    # similarity_matrix.to(device)
    # tensor = torch.tensor([temperature], device=device)
    # z_i_, z_j_ = representations[i], representations[j]
    param1 = 0.1
    param1 = torch.tensor([param1], device=device)
    sim_i_j = similarity_matrix[i, j]

    # calculate neg part
    start_index = batch_index * batch_size
    end_index = (batch_index + 1) * batch_size
    same_cluster_mask = torch.eq(cluster_labels[i], cluster_labels[start_index:end_index])
    # length of same_cluster_mask
    n = len(same_cluster_mask)
    # create all false tensor
    temp_false = torch.zeros(2 * n, dtype=torch.bool)
    temp_false[:n] = same_cluster_mask
    same_cluster_mask = temp_false.to(device)
    mask_neg = torch.ones_like(similarity_matrix).to(device) - \
               torch.matmul(same_cluster_mask.unsqueeze(0).int().float().t(),
                            same_cluster_mask.unsqueeze(0).int().float()) - \
               torch.eye(representations.shape[0]).to(device)
    mask_neg = mask_neg.clamp(min=0)

    # calculate fake_pos
    mask_fake_pos = torch.matmul(same_cluster_mask.unsqueeze(0).int().float().t(),
                                 same_cluster_mask.unsqueeze(0).int().float())
    # exclude itself
    mask_fake_pos[i] = 0

    mask_neg = mask_neg.to(device)
    mask_fake_pos = mask_fake_pos.to(device)

    # mask_neg = mask_neg.bool()
    # mask_fake_pos = mask_fake_pos.bool()
    # calculate numerator
    numerator = torch.sum(
        mask_fake_pos * (torch.exp(similarity_matrix[i, :] / temperature)).to(device)
    ) / param1 + torch.exp(sim_i_j / temperature)

    denominator = torch.sum(
        mask_neg * (torch.exp(similarity_matrix[i, :] / temperature)).to(device)
    )

    # calculate loss
    loss_ij = -torch.log(numerator / denominator)

    return loss_ij.squeeze(0)


def my_loss(model, train_features, augmented_features, cluster_labels, batch_index, temperature):
    cluster_labels = torch.from_numpy(cluster_labels).to(device)

    z_i = F.normalize(train_features, dim=1)
    z_j = F.normalize(augmented_features, dim=1)

    representations = torch.cat([z_i, z_j], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    batch_size = z_i.size(0)

    similarity_matrix = similarity_matrix.to(device)
    temperature = torch.tensor([temperature], device=device)
    representations = representations.to(device)

    N = batch_size
    loss = 0.0
    for k in range(0, N):
        loss += l_ij(k, k + N, representations, similarity_matrix, temperature, cluster_labels, batch_size, batch_index)
        # loss += simclr_loss(k, k + N, representations, similarity_matrix, temperature, train_features, batch_size)

    # add L2
    weight_decay = 0.1
    l2_reg = None
    for param in model.parameters():
        if l2_reg is None:
            l2_reg = param.norm(2)
        else:
            l2_reg = l2_reg + param.norm(2)

    loss = loss + weight_decay * l2_reg

    return 1.0 / N * loss


def simclr_loss(i, j, representations, similarity_matrix, temperature, emb_i, batch_size):
    z_i_, z_j_ = representations[i], representations[j]
    sim_i_j = similarity_matrix[i, j]

    numerator = torch.exp(sim_i_j / temperature)
    one_for_not_i = torch.ones((2 * batch_size,)).to(emb_i.device).scatter_(0, torch.tensor([i]).to(emb_i.device), 0.0)

    denominator = torch.sum(
        one_for_not_i * torch.exp(similarity_matrix[i, :] / temperature)
    )

    loss_ij = -torch.log(numerator / denominator)

    return loss_ij.squeeze(0)


def saveModelCheckpoints(model, epoch, optimizer, loss, accuracy):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    now = datetime.datetime.now()
    time_string = now.strftime("%m%d%H%M")
    PATH = './checkpoints/emb_' + time_string + '_' + str(epoch + 1) + '.pth'
    torch.save(checkpoint, PATH)


# def saveModel(model, epoch, ):
#     # 在每个epoch结束后保存checkpoint
#     checkpoint_path = "checkpoint.pth"
#     torch.save({
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': loss,
#         'accuracy': accuracy,
#     }, checkpoint_path)


def test(model, test_loader):
    model.eval()
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            _, _, logits = model(x)
            scores = F.softmax(logits, dim=1)
            predictions = torch.argmax(scores, dim=1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        accuracy = float(num_correct) / float(num_samples)
        print(f"Test accuracy: {accuracy:.4f}")
        return accuracy


def train(model, train_loader, augmented_train_loader, test_loader, cluster_labels, learning_rate, temperature,
          num_epoch, checkpoints):
    temperature = torch.tensor([temperature], device=device)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = RMSprop(model.parameters(), lr=learning_rate)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    checkpoint_path = checkpoints
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print("Successfully loaded checkpoint from '{}' (epoch {})".format(checkpoint_path, start_epoch))

    print("start trainning")
    for epoch in range(num_epoch):
        model.train()

        losses = 0.0
        accuracy = []
        check_loss = 0.0
        check_acc = 0.0
        loop = tqdm(enumerate(zip(train_loader, augmented_train_loader)), total=len(train_loader), miniters=10)
        for batch_index, (train_data, augmented_train_data) in loop:
            input1, labels = train_data[0].to(device), train_data[1].to(device)
            input2, labels = augmented_train_data[0].to(device), augmented_train_data[1].to(device)

            embX, train_features, logits_x = model.forward(input1)
            embY, augmented_features, logits_y = model.forward(input2)

            probs_x = F.softmax(logits_x, dim=1)
            probs_y = F.softmax(logits_y, dim=1)
            probs = probs_x + probs_y

            optimizer.zero_grad()
            loss = my_loss(model, train_features, augmented_features, cluster_labels, batch_index, temperature).to(
                device) + (criterion(logits_x, labels) + criterion(logits_y, labels)) / 2

            losses += loss.item()
            check_loss = loss.item()

            loss.backward()
            optimizer.step()

            predictions = torch.argmax(probs, dim=1)

            num_correct = (predictions == labels).sum()
            running_train_acc = float(num_correct) / float(input1.shape[0])
            check_acc = running_train_acc
            accuracy.append(running_train_acc)
            writer.add_scalar('Training loss', loss, global_step=epoch * len(train_loader) + batch_index)
            writer.add_scalar('Training accuracy', running_train_acc,
                              global_step=epoch * len(train_loader) + batch_index)
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

        best_acc = 0.0
        test_acc = test(model, test_loader)
        # 保存最好的模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')

        saveModelCheckpoints(model, epoch, optimizer, check_loss, check_acc)
    print('finish training')


def main():
    # train_loader, train_loader1 = get_data_loaders1(dataset_name, batch_size, image_size),
    train_loader, test_loader, augmented_train_loader = get_data_loaders()

    # pretrained model for cluster
    # model1 = EfficientNet.from_pretrained("efficientnet-b0")
    model1 = torchvision.models.resnet18(pretrained=True)
    model1.fc = torch.nn.Identity()
    model1.eval()
    # get extract_features
    if os.path.exists('./data/train_features.pt'):
        # 文件存在的情况下执行的代码
        print("features loaded")
        train_features = torch.load('./data/train_features.pt')
        train_labels = torch.load('./data/train_labels.pt')
    else:
        train_features, train_labels = extract_features(train_loader, model1)
        # test_features, test_labels = extract_features(testloader, model)

        # 保存特征向量
        torch.save(train_features, './data/train_features.pt')
        torch.save(train_labels, './data/train_labels.pt')

    if os.path.exists('./data/cluster_labels.txt'):
        # 文件存在的情况下执行的代码
        print("cluster loaded")
        cluster_labels = np.loadtxt('./data/cluster_labels.txt', dtype=int)
    else:
        num_clusters = 100
        cluster_labels, cluster_centers = first_clustering(num_clusters, train_features)
        # 将聚类标签保存到文件中
        np.savetxt('./data/cluster_labels.txt', cluster_labels, fmt='%d')

    learning_rate = 0.0005
    temperature = 0.05
    num_epochs = 5
    model2 = ImageEmbeddingClass()
    checkpoints = './checkpoints/emb_04252124_00.pth'

    # get all embeddings for training classifier
    train(model2, train_loader, augmented_train_loader, test_loader, cluster_labels, learning_rate, temperature,
          num_epochs, checkpoints)


if __name__ == '__main__':
    main()
