from efficientnet_pytorch import EfficientNet
import torch
from torch import nn


class ImageEmbeddingClass(nn.Module):
    class Identity(nn.Module):
        def __init__(self): super().__init__()

        def forward(self, x):
            return x

    def __init__(self, embedding_size=1024, num_classes=100):
        super().__init__()

        base_model = EfficientNet.from_pretrained("efficientnet-b0")
        internal_embedding_size = base_model._fc.in_features
        base_model._fc = ImageEmbeddingClass.Identity()

        self.embedding = base_model

        self.projection = nn.Sequential(
            nn.Linear(in_features=internal_embedding_size, out_features=embedding_size),
            nn.ReLU(),
            nn.Linear(in_features=embedding_size, out_features=embedding_size)
        )

        self.classifier = nn.Linear(embedding_size, num_classes)

    def calculate_embedding(self, image):
        return self.embedding(image)

    def forward(self, X):
        image = X
        embedding = self.calculate_embedding(image)
        projection = self.projection(embedding)
        class_logits = self.classifier(projection)
        return embedding, projection, class_logits
