import os
import pandas as pd
import matplotlib
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
import torch.multiprocessing
import logging
import datetime
import torch.optim as optim
import numpy as np
import yaml
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from hccl.cifar100_emb_train import get_data_loaders
from hccl.imageNet_dog_emb_train import get_dogdataset_loader
from simclr1.model.EmbModel import ImageEmbeddingClass

torch.multiprocessing.set_sharing_strategy('file_system')
matplotlib.use('Agg')
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# 创建 SummaryWriter 对象
writer = SummaryWriter()


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


def my_loss(model, train_features, augmented_features, batch_index, temperature):

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
        loss += simclr_loss(k, k + N, representations, similarity_matrix, temperature, train_features, batch_size)+ \
                simclr_loss(k + N, k, representations, similarity_matrix, temperature, train_features, batch_size)

    # # add L2
    # weight_decay = 0.1
    # l2_reg = None
    # for param in model.parameters():
    #     if l2_reg is None:
    #         l2_reg = param.norm(2)
    #     else:
    #         l2_reg = l2_reg + param.norm(2)
    #
    # loss = loss + weight_decay * l2_reg

    # return 1.0 / N * loss
    return 1.0 / (2 * N) * loss


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


def saveModelCheckpoints(model, epoch):
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict()
    }
    now = datetime.datetime.now()
    time_string = now.strftime("%m%d")
    PATH = './checkpoints/emb_' + time_string + '_' + str(epoch + 1) + '.pth'
    torch.save(checkpoint, PATH)


def train(model, train_loader, augmented_train_loader,  learning_rate, temperature, num_epoch, checkpoints):
    temperature = torch.tensor([temperature], device=device)
    model = model.to(device)

    # optimizer = RMSprop(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # checkpoint_path = checkpoints
    # if os.path.exists(checkpoint_path):
    #     checkpoint = torch.load(checkpoint_path)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     start_epoch = checkpoint['epoch']
    #     best_acc = checkpoint['best_acc']
    #     print("Successfully loaded checkpoint from '{}' (epoch {})".format(checkpoint_path, start_epoch))

    print("start trainning")
    for epoch in range(num_epoch):
        print('epoch[%d] start' % (epoch + 1))

        running_loss = 0.000
        loop = tqdm(enumerate(zip(train_loader, augmented_train_loader)), total=len(train_loader), miniters=10)
        for batch_index, (train_data, augmented_train_data) in loop:
            input1, labels = train_data[0].to(device), train_data[1].to(device)
            input2, labels = augmented_train_data[0].to(device), augmented_train_data[1].to(device)

            optimizer.zero_grad()

            train_features = model(input1)
            augmented_features = model(input2)

            loss = my_loss(model, train_features, augmented_features, batch_index, temperature).to(
                device)
            # (criterion(logits_x, labels) + criterion(logits_y, labels)) / 2

            loss.backward()
            optimizer.step()
            writer.add_scalar('Training loss', loss, global_step=epoch * len(train_loader) + batch_index)
            loop.set_description(f'Epoch [{epoch}/{num_epoch}]')
            loop.set_postfix(loss=loss.item())
            check_loss = loss.item()
            running_loss += loss.item()
            if batch_index % 30 == 29:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_index + 1, running_loss / 30))
                running_loss = 0.0
        # clear GPU cache
        # torch.cuda.empty_cache()

        # clear Python memory garbage
        # gc.collect()

        saveModelCheckpoints(model, epoch)
    print('finish training')


def main():
    # train_loader, train_loader1 = get_data_loaders1(dataset_name, batch_size, image_size),
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    dataset = config["dataset"]
    # train_loader, test_loader, augmented_train_loader, train_dataset = get_data_loaders(dataset)
    data_dir = '../hccl/data'
    df = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
    train_loader, augmented_train_loader, test_loader = get_dogdataset_loader(data_dir, df)
    emb_learning_rate = config["emb_learning_rate"]
    temperature = config["temperature"]
    num_epochs = 5
    model2 = ImageEmbeddingClass()
    checkpoints = './checkpoints/emb_04270911_00.pth'

    # get all embeddings for training classifier
    train(model2, train_loader, augmented_train_loader, emb_learning_rate, temperature, num_epochs, checkpoints)


if __name__ == '__main__':
    main()
