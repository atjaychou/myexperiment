import torch
from torch import nn
import torch.nn.functional as F
# from torchvision.datasets import STL10
import torchvision.transforms.functional as tvf
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, SequentialSampler
import random
# from efficientnet_pytorch import EfficientNet
from argparse import Namespace
from torch.multiprocessing import cpu_count
from torch.optim import RMSprop
# import pytorch_lightning as pl
# from pytorch_lightning.loggers import WandbLogger


def loss_simclr(emb_i, emb_j, batch_size, temperature, verbose):
    temperature = torch.tensor(temperature)
    """
    emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
    z_i, z_j as per SimCLR paper
    """
    z_i = F.normalize(emb_i, dim=1)
    z_j = F.normalize(emb_j, dim=1)

    representations = torch.cat([z_i, z_j], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
    if verbose: print("Similarity matrix\n", similarity_matrix, "\n")

    def l_ij(i, j):
        z_i_, z_j_ = representations[i], representations[j]
        sim_i_j = similarity_matrix[i, j]
        if verbose: print(f"sim({i}, {j})={sim_i_j}")

        numerator = torch.exp(sim_i_j / temperature)
        one_for_not_i = torch.ones((2 * batch_size,)).to(emb_i.device).scatter_(0, torch.tensor([i]), 0.0)
        if verbose: print(f"1{{k!={i}}}", one_for_not_i)

        denominator = torch.sum(
            one_for_not_i * torch.exp(similarity_matrix[i, :] / temperature)
        )
        if verbose: print("Denominator", denominator)

        loss_ij = -torch.log(numerator / denominator)
        if verbose: print(f"loss({i},{j})={loss_ij}\n")

        return loss_ij.squeeze(0)

    N = batch_size
    loss = 0.0
    for k in range(0, N):
        loss += l_ij(k, k + N) + l_ij(k + N, k)
    return 1.0 / (2 * N) * loss

class ContrastiveLossELI5(nn.Module):
    def __init__(self, batch_size, temperature=0.5, verbose=True):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.verbose = verbose

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        if self.verbose: print("Similarity matrix\n", similarity_matrix, "\n")

        def l_ij(i, j):
            z_i_, z_j_ = representations[i], representations[j]
            sim_i_j = similarity_matrix[i, j]
            if self.verbose: print(f"sim({i}, {j})={sim_i_j}")

            numerator = torch.exp(sim_i_j / self.temperature)
            one_for_not_i = torch.ones((2 * self.batch_size,)).to(emb_i.device).scatter_(0, torch.tensor([i]), 0.0)
            if self.verbose: print(f"1{{k!={i}}}", one_for_not_i)

            denominator = torch.sum(
                one_for_not_i * torch.exp(similarity_matrix[i, :] / self.temperature)
            )
            if self.verbose: print("Denominator", denominator)

            loss_ij = -torch.log(numerator / denominator)
            if self.verbose: print(f"loss({i},{j})={loss_ij}\n")

            return loss_ij.squeeze(0)

        N = self.batch_size
        loss = 0.0
        for k in range(0, N):
            loss += l_ij(k, k + N) + l_ij(k + N, k)
        return 1.0 / (2 * N) * loss


def random_rotate(image):
    if random.random() > 0.5:
        return tvf.rotate(image, angle=random.choice((0, 90, 180, 270)))
    return image


class ResizedRotation():
    def __init__(self, angle, output_size=(96, 96)):
        self.angle = angle
        self.output_size = output_size

    def angle_to_rad(self, ang): return np.pi * ang / 180.0

    def __call__(self, image):
        w, h = image.size
        new_h = int(
            np.abs(w * np.sin(self.angle_to_rad(90 - self.angle))) + np.abs(h * np.sin(self.angle_to_rad(self.angle))))
        new_w = int(
            np.abs(h * np.sin(self.angle_to_rad(90 - self.angle))) + np.abs(w * np.sin(self.angle_to_rad(self.angle))))
        img = tvf.resize(image, (new_w, new_h))
        img = tvf.rotate(img, self.angle)
        img = tvf.center_crop(img, self.output_size)
        return img



class WrapWithRandomParams():
    def __init__(self, constructor, ranges):
        self.constructor = constructor
        self.ranges = ranges

    def __call__(self, image):
        randoms = [float(np.random.uniform(low, high)) for _, (low, high) in zip(range(len(self.ranges)), self.ranges)]
        return self.constructor(*randoms)(image)


# stl10_unlabeled = STL10(".", split="unlabeled", download=True)


class PretrainingDatasetWrapper(Dataset):
    def __init__(self, ds: Dataset, target_size=(96, 96), debug=False):
        super().__init__()
        self.ds = ds
        self.debug = debug
        self.target_size = target_size
        if debug:
            print("DATASET IN DEBUG MODE")

        # I will be using network pre-trained on ImageNet first, which uses this normalization.
        # Remove this, if you're training from scratch or apply different transformations accordingly
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        random_resized_rotation = WrapWithRandomParams(lambda angle: ResizedRotation(angle, target_size),
                                                       [(0.0, 360.0)])
        self.randomize = transforms.Compose([
            transforms.RandomResizedCrop(target_size, scale=(1 / 3, 1.0), ratio=(0.3, 2.0)),
            transforms.RandomChoice([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Lambda(random_rotate)
            ]),
            transforms.RandomApply([
                random_resized_rotation
            ], p=0.33),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2)
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem_internal__(self, idx, preprocess=True):
        this_image_raw, _ = self.ds[idx]

        if self.debug:
            random.seed(idx)
            t1 = self.randomize(this_image_raw)
            random.seed(idx + 1)
            t2 = self.randomize(this_image_raw)
        else:
            t1 = self.randomize(this_image_raw)
            t2 = self.randomize(this_image_raw)

        if preprocess:
            t1 = self.preprocess(t1)
            t2 = self.preprocess(t2)
        else:
            t1 = transforms.ToTensor()(t1)
            t2 = transforms.ToTensor()(t2)

        return (t1, t2), torch.tensor(0)

    def __getitem__(self, idx):
        return self.__getitem_internal__(idx, True)

    def raw(self, idx):
        return self.__getitem_internal__(idx, False)


# ds = PretrainingDatasetWrapper(stl10_unlabeled, debug=False)


# class ImageEmbedding(nn.Module):
#     class Identity(nn.Module):
#         def __init__(self): super().__init__()
#
#         def forward(self, x):
#             return x
#
#     def __init__(self, embedding_size=1024):
#         super().__init__()
#
#         base_model = EfficientNet.from_pretrained("efficientnet-b0")
#         internal_embedding_size = base_model._fc.in_features
#         base_model._fc = ImageEmbedding.Identity()
#
#         self.embedding = base_model
#
#         self.projection = nn.Sequential(
#             nn.Linear(in_features=internal_embedding_size, out_features=embedding_size),
#             nn.ReLU(),
#             nn.Linear(in_features=embedding_size, out_features=embedding_size)
#         )
#
#     def calculate_embedding(self, image):
#         return self.embedding(image)
#
#     def forward(self, X):
#         image = X
#         embedding = self.calculate_embedding(image)
#         projection = self.projection(embedding)
#         return embedding, projection
#
#         (X, Y), y = batch
#         embX, projectionX = self.forward(X)
#         embY, projectionY = self.forward(Y)
#         loss = self.loss(projectionX, projectionY)
#
#
# class ImageEmbeddingModule(pl.LightningModule):
#     def __init__(self, hparams):
#         hparams = Namespace(**hparams) if isinstance(hparams, dict) else hparams
#         super().__init__()
#         self.hparams = hparams
#         self.model = ImageEmbedding()
#         self.loss = ContrastiveLoss(hparams.batch_size)
#
#     def total_steps(self):
#         return len(self.train_dataloader()) // self.hparams.epochs
#
#     def train_dataloader(self):
#         return DataLoader(PretrainingDatasetWrapper(stl10_unlabeled,
#                                                     debug=getattr(self.hparams, "debug", False)),
#                           batch_size=self.hparams.batch_size,
#                           num_workers=cpu_count(),
#                           sampler=SubsetRandomSampler(list(range(hparams.train_size))),
#                           drop_last=True)
#
#     def val_dataloader(self):
#         return DataLoader(PretrainingDatasetWrapper(stl10_unlabeled,
#                                                     debug=getattr(self.hparams, "debug", False)),
#                           batch_size=self.hparams.batch_size,
#                           shuffle=False,
#                           num_workers=cpu_count(),
#                           sampler=SequentialSampler(
#                               list(range(hparams.train_size + 1, hparams.train_size + hparams.validation_size))),
#                           drop_last=True)
#
#     def forward(self, X):
#         return self.model(X)
#
#     def step(self, batch, step_name="train"):
#         (X, Y), y = batch
#         embX, projectionX = self.forward(X)
#         embY, projectionY = self.forward(Y)
#         loss = self.loss(projectionX, projectionY)
#         loss_key = f"{step_name}_loss"
#         tensorboard_logs = {loss_key: loss}
#
#         return {("loss" if step_name == "train" else loss_key): loss, 'log': tensorboard_logs,
#                 "progress_bar": {loss_key: loss}}
#
#     def training_step(self, batch, batch_idx):
#         return self.step(batch, "train")
#
#     def validation_step(self, batch, batch_idx):
#         return self.step(batch, "val")
#
#     def validation_end(self, outputs):
#         if len(outputs) == 0:
#             return {"val_loss": torch.tensor(0)}
#         else:
#             loss = torch.stack([x["val_loss"] for x in outputs]).mean()
#             return {"val_loss": loss, "log": {"val_loss": loss}}
#
#     def configure_optimizers(self):
#         optimizer = RMSprop(self.model.parameters(), lr=self.hparams.lr)
#         return [optimizer], []
#
#
# hparams = Namespace(
#     lr=1e-3,
#     epochs=50,
#     batch_size=160,
#     train_size=10000,
#     validation_size=1000
# )
#
# module = ImageEmbeddingModule(hparams)
# t = pl.Trainer(gpus=1)
# lr_finder = t.lr_find(module)
#
# hparams = Namespace(
#     lr=0.000630957344480193,
#     epochs=10,
#     batch_size=160,
#     train_size=20000,
#     validation_size=1000
# )
# module = ImageEmbeddingModule(hparams)
# logger = WandbLogger(project="simclr-blogpost")
# logger.watch(module, log="all", log_freq=50)
# trainer = pl.Trainer(gpus=1, logger=logger)
# trainer.fit(module)
