import os
import matplotlib
import random
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

from hccl.model.EmbModel import ImageEmbeddingClass
from hccl.voc.get_test import VOCTestDataset
from hccl.voc.main import VOCClassificationDataset

torch.multiprocessing.set_sharing_strategy('file_system')
matplotlib.use('Agg')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 创建 SummaryWriter 对象
writer = SummaryWriter()


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

def get_data_loaders(dataset_dir, target_size=96):
    batch_size = 128
    num_clusters = 0
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                               std=[0.2675, 0.2565, 0.2761])])
    test_transform = transforms.Compose([transforms.Resize(size=(256, 256)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                              std=[0.2675, 0.2565, 0.2761])])
    # set log level as WARNING
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    dataset = VOCClassificationDataset(dataset_dir, transform=train_transform)
    augmented_trainset = VOCClassificationDataset(dataset_dir, transform=data_augmentation(96))
    testData = VOCTestDataset(dataset_dir, transform=test_transform)
    # 创建数据加载器
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    augmented_train_loader = DataLoader(augmented_trainset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(testData, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, augmented_train_loader, dataset

def l_ij(i, j, representations, similarity_matrix, temperature, cluster_labels, batch_size, batch_index, param1):
    # similarity_matrix.to(device)
    # tensor = torch.tensor([temperature], device=device)
    # z_i_, z_j_ = representations[i], representations[j]
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
    numerator = torch.exp(sim_i_j / temperature)

    denominator =  torch.sum(
        mask_neg * (torch.exp(similarity_matrix[i, :] / temperature)).to(device)
    ) * (1 - param1) + torch.sum(
        mask_fake_pos * (torch.exp(similarity_matrix[i, :] / temperature)).to(device)
    ) * param1

    # calculate loss
    loss_ij = -torch.log(numerator / denominator)

    return loss_ij.squeeze(0)


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

def D(p, z, version='simplified'):  # negative cosine similarity
    if version == 'original':
        z = z.detach()  # stop gradient
        p = F.normalize(p, dim=1)  # l2-normalize
        z = F.normalize(z, dim=1)  # l2-normalize
        return -(p * z).sum(dim=1).mean()

    elif version == 'simplified':  # same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception

def my_loss(train_features, augmented_features, cluster_train_labels, cluster_aug_labels, batch_index, temperature, param1):
    cluster_train_labels = torch.from_numpy(cluster_train_labels).to(device)
    cluster_aug_labels = torch.from_numpy(cluster_aug_labels).to(device)

    z_i = F.normalize(train_features, dim=1)
    z_j = F.normalize(augmented_features, dim=1)

    representations = torch.cat([z_i, z_j], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
    # similarity_matrix = F.pairwise_distance(representations.unsqueeze(1), representations.unsqueeze(0), p=2)
    # similarity_matrix = torch.cdist(representations.unsqueeze(1), representations.unsqueeze(0))
    # similarity_matrix = similarity_matrix.reshape(256, 256)

    batch_size = z_i.size(0)

    similarity_matrix = similarity_matrix.to(device)
    temperature = torch.tensor([temperature], device=device)
    representations = representations.to(device)

    N = batch_size
    loss = 0.0
    for k in range(0, N):
        # loss += (l_ij(k, k + N, representations, similarity_matrix, temperature, cluster_labels, batch_size,
        #               batch_index)
        #          + simclr_loss(k + N, k, representations, similarity_matrix, temperature, train_features, batch_size))
        loss += (l_ij(k, k + N, representations, similarity_matrix, temperature, cluster_train_labels, batch_size, batch_index,param1) +
                 l_ij(k+N, k, representations, similarity_matrix, temperature, cluster_aug_labels, batch_size, batch_index,param1))
        # 只考虑对一个训练集聚类
        # loss += l_ij(k, k + N, representations, similarity_matrix, temperature, cluster_train_labels, batch_size,
        #               batch_index)

    # add L2
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
    return 1.0 / (2 *N) * loss
    # return 1.0 / (3 * N) * loss


def saveModelCheckpoints(model, epoch):
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict()
    }
    now = datetime.datetime.now()
    time_string = now.strftime("%m%d%H%M")
    PATH = './checkpoints/emb_' + time_string + '_' + str(epoch + 1) + '.pth'
    torch.save(checkpoint, PATH)


def train(model, train_loader, augmented_train_loader, cluster_list, learning_rate, temperature,
          num_epoch, checkpoints):
    param1 = 0.6
    temperature = torch.tensor([temperature], device=device)
    param1 = torch.tensor([param1], device=device)
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

            train_layer1_projection = model(input1)
            augmented_layer1_projection = model(input2)

            loss = (my_loss(train_layer1_projection, augmented_layer1_projection, cluster_list[1], cluster_list[3], batch_index, temperature, param1)).to(device)


            loss.backward()
            optimizer.step()
            writer.add_scalar('Training loss', loss, global_step=epoch * len(train_loader) + batch_index)
            loop.set_description(f'Epoch [{epoch + 1}/{num_epoch}]')
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
    # 设置随机数种子
    setup_seed(100)
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    # train_loader, train_loader1 = get_data_loaders1(dataset_name, batch_size, image_size),
    train_loader, test_loader, augmented_train_loader, _ = get_data_loaders(dataset_dir = './data/VOC')

    cluster_train_labels = np.loadtxt('./cluster/data/cluster_labels_voc_505.txt', dtype=int)
    cluster_aug_labels = np.loadtxt('./cluster/data/cluster_labels_voc_aug_505.txt', dtype=int)

    cluster_train_labels1 = [item[0] for item in cluster_train_labels]
    cluster_train_labels1 = np.array(cluster_train_labels1)
    cluster_train_labels2 = [item[1] for item in cluster_train_labels]
    cluster_train_labels2 = np.array(cluster_train_labels2)


    cluster_aug_labels1 = [item[0] for item in cluster_aug_labels]
    cluster_aug_labels1 = np.array(cluster_aug_labels1)
    cluster_aug_labels2 = [item[1] for item in cluster_aug_labels]
    cluster_aug_labels2 = np.array(cluster_aug_labels2)

    cluster_list = [cluster_train_labels1, cluster_train_labels2, cluster_aug_labels1, cluster_aug_labels2]



    emb_learning_rate = config["emb_learning_rate"]
    temperature = config["temperature"]
    num_epochs = 5
    model2 = ImageEmbeddingClass()
    checkpoints = './checkpoints/emb_04270911_00.pth'

    # get all embeddings for training classifier
    train(model2, train_loader, augmented_train_loader, cluster_list, emb_learning_rate,
          temperature, num_epochs, checkpoints)


if __name__ == '__main__':
    main()
