import torch
import torch.nn as nn

from simsiam.model.simsiam import SimSiam


class SimSiamClassifier(nn.Module):
    def __init__(self, n_classes, freeze_base, checkpath):
        super().__init__()

        base_model = SimSiam()
        checkpoint = torch.load(checkpath)
        base_model.load_state_dict(checkpoint['state_dict'])

        self.embeddings = base_model.backbone

        if freeze_base:
            print("Freezing embeddings")
            for param in self.embeddings.parameters():
                param.requires_grad = False

        # Only linear projection on top of the embeddings should be enough
        self.classifier = nn.Linear(in_features=base_model.encoder[0]._fc.out_features,
                                    out_features=n_classes if n_classes > 2 else 1)

    def forward(self, X, *args):
        emb = self.embeddings(X)
        return self.classifier(emb)
