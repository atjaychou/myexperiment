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
from cluster.ImageEmb import ImageEmbedding
import datetime
from torch.optim import RMSprop
import torch.optim as optim

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
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def get_data_loaders(target_size=96):  # target_size=(96, 96)
    # set log level as WARNING
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    data_dir = '../data'
    download = False

    if not os.path.exists(data_dir):
        download = True

    seed = 2245
    # CIFAR-100 dataset
    train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=download, transform=transforms.ToTensor())
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=download, transform=transforms.ToTensor())

    batch_size = 64

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

    # calculate numerator
    numerator = torch.exp(sim_i_j / temperature)

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

    denominator = param1 * torch.sum(
        mask_fake_pos * (torch.exp(similarity_matrix[i, :] / temperature)).to(device)
    ) + (1 - param1) * torch.sum(
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
        loss = 0.0
        # loss += l_ij(k, k + N, representations, similarity_matrix, temperature, cluster_labels, batch_size, batch_index)
        loss += simclr_loss(k, k + N, representations, similarity_matrix, temperature, train_features, batch_size)

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
    one_for_not_i = torch.ones((2 * batch_size,)).to(emb_i.device).scatter_(0, torch.tensor([i]), 0.0)

    denominator = torch.sum(
        one_for_not_i * torch.exp(similarity_matrix[i, :] / temperature)
    )

    loss_ij = -torch.log(numerator / denominator)

    return loss_ij.squeeze(0)


def train(model, train_loader, augmented_train_loader, cluster_labels, learning_rate, temperature, num_epochs):
    batch_num = len(train_loader)
    model = model.to(device)
    print("start trainning")
    for epoch in range(num_epochs):
        model.train()
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        optimizer = RMSprop(model.parameters(), lr=learning_rate)

        running_loss = 0.0
        for batch_index, (train_data, augmented_train_data) in enumerate(
                zip(train_loader, augmented_train_loader)):
            input1, labels = train_data[0].to(device), train_data[1].to(device)
            input2, labels = augmented_train_data[0].to(device), augmented_train_data[1].to(device)

            embX, train_features = model.forward(input1)
            embY, augmented_features = model.forward(input2)

            optimizer.zero_grad()
            loss = my_loss(model, train_features, augmented_features, cluster_labels, batch_index, temperature).to(
                device)

            loss = loss.requires_grad_()

            loss.backward()
            optimizer.step()

            running_loss += loss.detach().item()

            if batch_index % 50 == 49:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_index + 1, running_loss / 50))
                running_loss = 0.0
        print('[Epoch %d] Loss: %.4f' % (epoch + 1, running_loss / batch_num))
        # save model
        now = datetime.datetime.now()
        time_string = now.strftime("%m%d%H%M")
        PATH = './emb_' + time_string + '_' + str(epoch + 1) + '.pth'
        torch.save(model.state_dict(), PATH)
    print('finish training')


def main():
    # train_loader, train_loader1 = get_data_loaders1(dataset_name, batch_size, image_size),
    train_loader, test_loader, augmented_train_loader = get_data_loaders()

    # pretrained model for cluster
    # model1 = EfficientNet.from_pretrained("efficientnet-b0")
    model1 = torchvision.models.resnet18(pretrained=True)
    model1.fc = torch.nn.Identity()
    # get extract_features
    train_features, train_labels = extract_features(train_loader, model1)
    # augmented_features, augmented_labels = extract_features(augmented_train_loader, model1)

    num_clusters = 100
    cluster_labels, cluster_centers = first_clustering(num_clusters, train_features)

    learning_rate = 0.0006
    temperature = 0.5
    num_epochs = 10
    model2 = ImageEmbedding()

    # get all embeddings for training classifier
    train(model2, train_loader, augmented_train_loader, cluster_labels, learning_rate, temperature, num_epochs)


if __name__ == '__main__':
    main()
