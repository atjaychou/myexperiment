import torch
from hccl.model.resnet50 import ImageEmbeddingClass
import torch.nn as nn

class SimCLRClassifier(nn.Module):
    def __init__(self, n_classes, freeze_base, checkpath):
        super().__init__()

        base_model = ImageEmbeddingClass()
        checkpoint = torch.load(checkpath)
        base_model.load_state_dict(checkpoint['state_dict'])

        self.embeddings = base_model.embedding
        # self.projection = base_model.projection1

        if freeze_base:
            print("Freezing embeddings")
            for param in self.embeddings.parameters():
                param.requires_grad = False

        # Only linear projection on top of the embeddings should be enough
        self.classifier = nn.Linear(in_features=base_model.projection1[0].in_features,
                                    out_features=n_classes if n_classes > 2 else 1)

    def forward(self, X, *args):
        emb = self.embeddings(X)
        # pro = self.projection(emb)
        return self.classifier(emb)
