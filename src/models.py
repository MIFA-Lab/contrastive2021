import torch
import torch.nn as nn
import pytorch_lightning as pl
import lightly
import torch.nn.functional as F
from src.util import knn_predict

knn_k = 200
knn_t = 0.1
classes = 10


class BenchmarkModule(pl.LightningModule):
    """A PyTorch Lightning Module for automated kNN callback
    At the end of every training epoch we create a feature bank by inferencing
    the backbone on the dataloader passed to the module. 
    At every validation step we predict features on the validation data.
    After all predictions on validation data (validation_epoch_end) we evaluate
    the predictions on a kNN classifier on the validation data using the 
    feature_bank features from the train data.
    We can access the highest accuracy during a kNN prediction using the 
    max_accuracy attribute.
    """

    def __init__(self, dataloader_kNN, epochs):
        super().__init__()
        self.backbone = nn.Module()
        self.max_accuracy = 0.0
        self.dataloader_kNN = dataloader_kNN
        self.epochs = epochs

    def training_epoch_end(self, outputs):
        # print losses
        losses = [i['loss'].item() for i in outputs]
        loss_avg = sum(losses)/len(losses)
        print(f'Epoch {self.current_epoch+1}/{self.epochs}: train loss = {loss_avg:.2f}')

        # update feature bank at the end of each training epoch
        self.backbone.eval()
        self.feature_bank = []
        self.targets_bank = []
        with torch.no_grad():
            for data in self.dataloader_kNN:
                img, target, _ = data
                if torch.cuda.is_available():
                    img = img.cuda()
                    target = target.cuda()
                feature = self.backbone(img).squeeze()
                feature = F.normalize(feature, dim=1)
                self.feature_bank.append(feature)
                self.targets_bank.append(target)
        self.feature_bank = torch.cat(
            self.feature_bank, dim=0).t().contiguous()
        self.targets_bank = torch.cat(
            self.targets_bank, dim=0).t().contiguous()
        self.backbone.train()

    def validation_step(self, batch, batch_idx):
        # we can only do kNN predictions once we have a feature bank
        if hasattr(self, 'feature_bank') and hasattr(self, 'targets_bank'):
            images, targets, _ = batch
            feature = self.backbone(images).squeeze()
            feature = F.normalize(feature, dim=1)
            pred_labels = knn_predict(
                feature, self.feature_bank, self.targets_bank, classes, knn_k, knn_t)
            num = images.size(0)
            top1 = (pred_labels[:, 0] == targets).float().sum().item()
            return (num, top1)

    def validation_epoch_end(self, outputs):
        if outputs:
            total_num = 0
            total_top1 = 0.
            for (num, top1) in outputs:
                total_num += num
                total_top1 += top1
            acc = float(total_top1 / total_num)
            if acc > self.max_accuracy:
                self.max_accuracy = acc
            print(f'Epoch {self.current_epoch+1}/{self.epochs}: KNN acc = {100*acc:.2f}')
            # self.log('kNN_accuracy', acc * 100.0, prog_bar=True)


class SimCLRModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, epochs):
        super().__init__(dataloader_kNN, epochs)
        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18')
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )
        # create a simclr model based on ResNet
        self.resnet_simclr = \
            lightly.models.SimCLR(self.backbone, num_ftrs=512)  # add a 2-layer projection head
        self.criterion = lightly.loss.NTXentLoss()

    def forward(self, x):
        self.resnet_simclr(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        x0, x1 = self.resnet_simclr(x0, x1)
        loss = self.criterion(x0, x1)
        # self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.resnet_simclr.parameters(),
            lr=6e-2,
            momentum=0.9,
            weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, self.epochs)
        return [optim], [scheduler]


class MocoModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, epochs, memory_bank_size=4096):
        super().__init__(dataloader_kNN, epochs)
        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18', num_splits=8)
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )
        # create a moco model based on ResNet
        self.resnet_moco = \
            lightly.models.MoCo(self.backbone, num_ftrs=512,
                                m=0.99, batch_shuffle=True)
        # create our loss with the optional memory bank
        self.criterion = lightly.loss.NTXentLoss(
            temperature=0.1,
            memory_bank_size=memory_bank_size)

    def forward(self, x):
        self.resnet_moco(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        # We use a symmetric loss (model trains faster at little compute overhead)
        # https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
        loss_1 = self.criterion(*self.resnet_moco(x0, x1))
        loss_2 = self.criterion(*self.resnet_moco(x1, x0))
        loss = 0.5 * (loss_1 + loss_2)
        # self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.resnet_moco.parameters(),
            lr=0.08,
            momentum=0.9,
            weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.epochs)
        return [optim], [scheduler]


class SimSiamModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, epochs):
        super().__init__(dataloader_kNN, epochs)
        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18')
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )
        # create a simsiam model based on ResNet
        self.resnet_simsiam = \
            lightly.models.SimSiam(
                self.backbone, num_ftrs=512, num_mlp_layers=2)
        self.criterion = lightly.loss.SymNegCosineSimilarityLoss()

    def forward(self, x):
        self.resnet_simsiam(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        x0, x1 = self.resnet_simsiam(x0, x1)
        loss = self.criterion(x0, x1)
        # self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_simsiam.parameters(), lr=0.05,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, self.epochs)
        return [optim], [scheduler]


class BarlowTwinsLoss(torch.nn.Module):
    def __init__(self, lambda_param=5e-3):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor):
        device = z_a.device
        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0)  # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0)  # NxD
        N = z_a.size(0)
        D = z_a.size(1)
        # cross-correlation matrix
        c = torch.mm(z_a_norm.T, z_b_norm) / N  # DxD
        # loss
        c_diff = (c - torch.eye(D, device=device)).pow(2)  # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param
        loss = c_diff.sum()
        return loss


class BarlowTwinsModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, epochs):
        super().__init__(dataloader_kNN, epochs)
        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18')
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )
        # create a simsiam model based on ResNet
        # note that BarlowTwins has the same architecture
        self.resnet_simsiam = \
            lightly.models.SimSiam(
                self.backbone, num_ftrs=512, num_mlp_layers=3)
        self.criterion = BarlowTwinsLoss(lambda_param=5e-3)

    def forward(self, x):
        self.resnet_simsiam(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        x0, x1 = self.resnet_simsiam(x0, x1)
        # our simsiam model returns both (features + projection head)
        z_a, _ = x0
        z_b, _ = x1
        loss = self.criterion(z_a, z_b)
        # self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_simsiam.parameters(), lr=0.11,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, self.epochs)
        return [optim], [scheduler]
