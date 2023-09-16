import torch
from cluster.ImageEmb import ImageEmbedding
import torch.nn as nn


class SimCLRClassifier(nn.Module):
    def __init__(self, n_classes, freeze_base, hidden_size=512):
        super().__init__()

        base_model = ImageEmbedding()
        base_model.load_state_dict(torch.load("emb_04242037_1.pth"))

        self.embeddings = base_model.embedding

        if freeze_base:
            print("Freezing embeddings")
            for param in self.embeddings.parameters():
                param.requires_grad = False

        # Only linear projection on top of the embeddings should be enough
        self.classifier = nn.Linear(in_features=base_model.projection[0].in_features,
                                    out_features=n_classes if n_classes > 2 else 1)

    def forward(self, X, *args):
        emb = self.embeddings(X)
        return self.classifier(emb)
