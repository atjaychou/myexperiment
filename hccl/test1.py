import numpy as np


cluster_train_labels = np.loadtxt('./data/cluster_labels_cifar100.txt', dtype=int)
cluster_aug_labels = np.loadtxt('./data/cluster_for_aug_labels_cifar100.txt', dtype=int)

cluster_train_labels1 = [item[0] for item in cluster_train_labels]
cluster_train_labels1 = np.array(cluster_train_labels1)
cluster_train_labels2 = [item[1] for item in cluster_train_labels]
cluster_train_labels2 = np.array(cluster_train_labels2)

cluster_aug_labels1 = [item[0] for item in cluster_aug_labels]
cluster_aug_labels1 = np.array(cluster_aug_labels1)
cluster_aug_labels2 = [item[1] for item in cluster_aug_labels]
cluster_aug_labels2 = np.array(cluster_aug_labels2)

cluster_list = [cluster_train_labels1, cluster_train_labels2, cluster_aug_labels1, cluster_aug_labels2]



print(11)