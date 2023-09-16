import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from hccl.cifar100_emb_train import setup_seed, train, get_cluster_list
from hccl.data.dataset import DogDataset, DogDatasetTest
from hccl.model.EmbModel import ImageEmbeddingClass


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

def get_dogdataset_loader(data_dir, df):

    breeds = df.breed.unique()  # 长度是120，即类别数
    breeds.sort()
    breed2idx = dict((breed, i) for i, breed in enumerate(breeds))
    df['label_idx'] = [breed2idx[b] for b in df.breed]

    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize(size=(256, 256)),
                                         transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                              std=[0.2675, 0.2565, 0.2761])])

    train_dataset = DogDataset(df, os.path.join(data_dir, 'train'), data_augmentation(96))
    test_dataset = DogDatasetTest(os.path.join(data_dir, 'test'), test_transform)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size, shuffle=False, drop_last=True)
    augmented_train_loader = DataLoader(train_dataset, batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, drop_last=False)

    return train_loader, augmented_train_loader, test_loader

def main():
    # 设置随机数种子
    setup_seed(100)
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)


    emb_learning_rate = config["emb_learning_rate"]
    temperature = config["temperature"]
    num_epochs = 5
    model2 = ImageEmbeddingClass()
    checkpoints = './checkpoints/emb_04270911_00.pth'

    data_dir = './data'
    df = pd.read_csv(os.path.join(data_dir, 'labels.csv'))

    train_loader, augmented_train_loader, test_loader = get_dogdataset_loader(data_dir, df)

    train_cluster_path = config["train_cluster_path"]
    aug_cluster_path = config["aug_cluster_path"]

    cluster_list = get_cluster_list(train_cluster_path, aug_cluster_path)

    # get all embeddings for training classifier
    train(model2, train_loader, augmented_train_loader, cluster_list, emb_learning_rate,
          temperature, num_epochs, checkpoints)


    print(11)



if __name__ == '__main__':
    main()