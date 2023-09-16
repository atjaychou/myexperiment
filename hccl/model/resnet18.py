import torch
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
import torch
from torch import nn



class ImageEmbeddingClass(nn.Module):
    class Identity(nn.Module):
        def __init__(self): super().__init__()

        def forward(self, x):
            return x

    def __init__(self, embedding_size=512, num_classes=100):
        super().__init__()

        base_model = models.resnet18(pretrained=True)
        for param in base_model.parameters():
            param.requires_grad = True
        internal_embedding_size = base_model.fc.in_features
        base_model.fc = ImageEmbeddingClass.Identity()

        self.embedding = base_model

        self.projection1 = nn.Sequential(
            nn.Linear(in_features=internal_embedding_size, out_features=embedding_size),
            nn.ReLU(),
            nn.Linear(in_features=embedding_size, out_features=embedding_size)
        )
        self.softmax = nn.Softmax(dim=1)

        # self.projection2 = nn.Sequential(
        #     nn.Linear(in_features=internal_embedding_size, out_features=embedding_size),
        #     nn.ReLU(),
        #     nn.Linear(in_features=embedding_size, out_features=embedding_size)
        # )

        # self.classifier = nn.Linear(embedding_size, num_classes)

    def calculate_embedding(self, image):
        return self.embedding(image)

    def forward(self, X):
        image = X
        embedding = self.calculate_embedding(image)
        projection1 = self.projection1(embedding)
        projection1 = self.softmax(projection1)

        # projection2 = self.projection2(embedding)

        # class_logits = self.classifier(projection)
        # return embedding, projection, class_logits
        # return projection1, projection2
        return projection1
