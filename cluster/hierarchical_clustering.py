import torch
import torchvision
from torchvision.transforms import transforms
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# Load CIFAR100 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.CIFAR100(root='../data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True, num_workers=2)

# Define the number of clusters at each level
num_clusters = [100, 50, 20, 10]

# Define the clustering model
class ClusteringModel:
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters
        self.kmeans = KMeans(n_clusters=num_clusters)

    def fit(self, x):
        x = x.view(x.size(0), -1).cpu().numpy()
        self.kmeans.fit(x)

    def predict(self, x):
        x = x.view(x.size(0), -1).cpu().numpy()
        return torch.from_numpy(self.kmeans.predict(x)).to(device)

# Move model and data to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True, num_workers=2, pin_memory=True)
models = [ClusteringModel(n) for n in num_clusters]

# Train the clustering model at each level
for i in range(len(num_clusters)):
    model = models[i]
    for images, _ in trainloader:
        images = images.to(device)
        model.fit(images)
    models[i] = model

# Perform hierarchical clustering
labels = None
for i in range(len(models)):
    if labels is None:
        # At the first level, cluster the original data
        labels = [models[i].predict(images) for images, _ in trainloader]
    else:
        # At subsequent levels, cluster the previous level's labels
        labels = [models[i].predict(l) for l in labels]

    # Evaluate the clustering accuracy
    true_labels = [labels.numpy().flatten() for _, labels in trainloader]
    true_labels = torch.cat(true_labels).to(device)
    labels = torch.cat(labels).to(device)
    accuracy = accuracy_score(true_labels.cpu(), labels.cpu())
    print(f'Level {i+1} clustering accuracy: {accuracy}')
