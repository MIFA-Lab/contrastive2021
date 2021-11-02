# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import pytorch_lightning as pl
import lightly

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model",  help="simclr, simsiam, twins, moco")
parser.add_argument("--epoch",  help="max_epochs", default="800")
parser.add_argument("--batch",  help="batch_size", default="512")
parser.add_argument("--runs",  help="n_runs", default="1")
parser.add_argument("--lamb",  help="lambda in twins", default="5e-3")
parser.add_argument("--strength",  help="color_strength", default="0.5")
parser.add_argument("--augs",  help="augmentation combinations: default, color, a, ab, abc, abcd, abcde", default="default")
parser.add_argument("--twinslr",  help="learning rate for twins", default="1e-3")
parser.add_argument("--dataset",  help="dataset directory", default="./CIFAR10/")
args = parser.parse_args()

num_workers = 8
memory_bank_size = 4096
# set max_epochs to 800 for long run (takes around 10h on a single V100)
max_epochs = int(args.epoch)
knn_k = 200
knn_t = 0.1
classes = 10
color_strength = float(args.strength)
twins_lr = float(args.twinslr)
dataset_dir = args.dataset

# benchmark
n_runs = int(args.runs) # optional, increase to create multiple runs and report mean + std
batch_size = int(args.batch)
logs_root_dir = os.path.join(os.getcwd(), 'results', args.model + '_' + args.epoch + '_' + args.batch + '_' + args.runs + '_' + args.augs)


# use a GPU if available
gpus = 1 if torch.cuda.is_available() else 0

if args.augs == 'default':
    # Use SimCLR augmentations, additionally, disable blur
    collate_fn = lightly.data.SimCLRCollateFunction(
        input_size = 32,
        gaussian_blur = 0.,
    )
elif args.augs == 'color':
    # Use SimCLR augmentations, additionally, disable blur
    collate_fn = lightly.data.SimCLRCollateFunction(
        input_size = 32,
        min_scale = 0.08,
        gaussian_blur = 0.,
        random_gray_scale = 0.,
        cj_prob = 0.8,
        cj_strength = color_strength,
        hf_prob = 0.,
    )
elif args.augs == 'a':
    collate_fn = lightly.data.SimCLRCollateFunction(
        input_size = 32,
        min_scale = 0.08,
        gaussian_blur = 0.,
        random_gray_scale = 0.,
        cj_prob = 0.,
        cj_strength = 0.5,
        hf_prob = 0.,
        # https://docs.lightly.ai/lightly.data.html#module-lightly.data.collate
    )
elif args.augs == 'ab':
    collate_fn = lightly.data.SimCLRCollateFunction(
        input_size = 32,
        min_scale = 0.08,
        gaussian_blur = 0.5,
        random_gray_scale = 0.,
        cj_prob = 0.,
        cj_strength = 0.5,
        hf_prob = 0.,
        # https://docs.lightly.ai/lightly.data.html#module-lightly.data.collate
    )
elif args.augs == 'abc':
    collate_fn = lightly.data.SimCLRCollateFunction(
        input_size = 32,
        min_scale = 0.08,
        gaussian_blur = 0.5,
        random_gray_scale = 0.2,
        cj_prob = 0.,
        cj_strength = 0.5,
        hf_prob = 0.,
        # https://docs.lightly.ai/lightly.data.html#module-lightly.data.collate
    )
elif args.augs == 'abcd':
    collate_fn = lightly.data.SimCLRCollateFunction(
        input_size = 32,
        min_scale = 0.08,
        gaussian_blur = 0.5,
        random_gray_scale = 0.2,
        cj_prob = 0.8,
        cj_strength = 0.5,
        hf_prob = 0.,
        # https://docs.lightly.ai/lightly.data.html#module-lightly.data.collate
    )
elif args.augs == 'abcde':
    collate_fn = lightly.data.SimCLRCollateFunction(
        input_size = 32,
        min_scale = 0.08,
        gaussian_blur = 0.5,
        random_gray_scale = 0.2,
        cj_prob = 0.8,
        cj_strength = 0.5,
        hf_prob = 0.5,
        # https://docs.lightly.ai/lightly.data.html#module-lightly.data.collate
    )
else:
    print('Augmentation setting is invalid!')

# No additional augmentations for the test set
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'],
    )
])

dataset_train_ssl = lightly.data.LightlyDataset.from_torch_dataset(
    torchvision.datasets.CIFAR10(root=dataset_dir,train=True,download=True))

# we use test transformations for getting the feature for kNN on train data
dataset_train_kNN = lightly.data.LightlyDataset.from_torch_dataset(
    torchvision.datasets.CIFAR10(root=dataset_dir,train=True,download=True,transform=test_transforms))

dataset_test = lightly.data.LightlyDataset.from_torch_dataset(
    torchvision.datasets.CIFAR10(root=dataset_dir,train=False,download=True,transform=test_transforms))

def get_data_loaders(batch_size: int):
    """Helper method to create dataloaders for ssl, kNN train and kNN test

    Args:
        batch_size: Desired batch size for all dataloaders
    """
    dataloader_train_ssl = torch.utils.data.DataLoader(
        dataset_train_ssl,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers
    )

    dataloader_train_kNN = torch.utils.data.DataLoader(
        dataset_train_kNN,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    return dataloader_train_ssl, dataloader_train_kNN, dataloader_test

# code from here:
# https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb

# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes: int, knn_k: int, knn_t: float):
    """Helper method to run kNN predictions on features based on a feature bank

    Args:
        feature: Tensor of shape [N, D] consisting of N D-dimensional features
        feature_bank: Tensor of a database of features used for kNN
        feature_labels: Labels for the features in our feature_bank
        classes: Number of classes (e.g. 10 for CIFAR-10)
        knn_k: Number of k neighbors used for kNN
        knn_t: 

    """

    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)

    # we do a reweighting of the similarities 
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


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
    def __init__(self, dataloader_kNN):
        super().__init__()
        self.backbone = nn.Module()
        self.max_accuracy = 0.0
        self.dataloader_kNN = dataloader_kNN

    def training_epoch_end(self, outputs):
        # update feature bank at the end of each training epoch
        self.backbone.eval()
        self.feature_bank = []
        self.targets_bank = []
        with torch.no_grad():
            for data in self.dataloader_kNN:
                img, target, _ = data
                if gpus > 0:
                    img = img.cuda()
                    target = target.cuda()
                feature = self.backbone(img).squeeze()
                feature = F.normalize(feature, dim=1)
                self.feature_bank.append(feature)
                self.targets_bank.append(target)
        self.feature_bank = torch.cat(self.feature_bank, dim=0).t().contiguous()
        self.targets_bank = torch.cat(self.targets_bank, dim=0).t().contiguous()
        self.backbone.train()

    def validation_step(self, batch, batch_idx):
        # we can only do kNN predictions once we have a feature bank
        if hasattr(self, 'feature_bank') and hasattr(self, 'targets_bank'):
            images, targets, _ = batch
            feature = self.backbone(images).squeeze()
            feature = F.normalize(feature, dim=1)
            pred_labels = knn_predict(feature, self.feature_bank, self.targets_bank, classes, knn_k, knn_t)
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
            self.log('kNN_accuracy', acc * 100.0, prog_bar=True)


class MocoModel(BenchmarkModule):
    def __init__(self, dataloader_kNN):
        super().__init__(dataloader_kNN)
        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18', num_splits=8)
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )
        # create a moco model based on ResNet
        self.resnet_moco = \
            lightly.models.MoCo(self.backbone, num_ftrs=512, m=0.99, batch_shuffle=True)
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
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_moco.parameters(), lr=6e-2,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


class SimCLRModel(BenchmarkModule):
    def __init__(self, dataloader_kNN):
        super().__init__(dataloader_kNN)
        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18')
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )
        # create a simclr model based on ResNet
        self.resnet_simclr = \
            lightly.models.SimCLR(self.backbone, num_ftrs=512)
        self.criterion = lightly.loss.NTXentLoss()
            
    def forward(self, x):
        self.resnet_simclr(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        x0, x1 = self.resnet_simclr(x0, x1)
        loss = self.criterion(x0, x1)
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_simclr.parameters(), lr=6e-2,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


class SimSiamModel(BenchmarkModule):
    def __init__(self, dataloader_kNN):
        super().__init__(dataloader_kNN)
        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18')
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )
        # create a simsiam model based on ResNet
        self.resnet_simsiam = \
            lightly.models.SimSiam(self.backbone, num_ftrs=512, num_mlp_layers=2)
        self.criterion = lightly.loss.SymNegCosineSimilarityLoss()
            
    def forward(self, x):
        self.resnet_simsiam(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        x0, x1 = self.resnet_simsiam(x0, x1)
        loss = self.criterion(x0, x1)
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_simsiam.parameters(), lr=6e-2,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


class BarlowTwinsLoss(torch.nn.Module):
    def __init__(self, lambda_param=5e-3):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor):
        device = z_a.device
        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0) # NxD
        N = z_a.size(0)
        D = z_a.size(1)
        # cross-correlation matrix
        c = torch.mm(z_a_norm.T, z_b_norm) / N # DxD
        # loss
        c_diff = (c - torch.eye(D, device=device)).pow(2) # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param
        loss = c_diff.sum()
        return loss

class BarlowTwinsModel(BenchmarkModule):
    def __init__(self, dataloader_kNN):
        super().__init__(dataloader_kNN)
        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18')
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )
        # create a simsiam model based on ResNet
        # note that BarlowTwins has the same architecture
        self.resnet_simsiam = \
            lightly.models.SimSiam(self.backbone, num_ftrs=512, num_mlp_layers=3)
        self.criterion = BarlowTwinsLoss(lambda_param=float(args.lamb))
            
    def forward(self, x):
        self.resnet_simsiam(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        x0, x1 = self.resnet_simsiam(x0, x1)
        # our simsiam model returns both (features + projection head)
        z_a, _ = x0
        z_b, _ = x1
        loss = self.criterion(z_a, z_b)
        self.log('train_loss_ssl', loss)
        return loss

    # learning rate warm-up
    def optimizer_steps(self,
                        epoch=None,
                        batch_idx=None,
                        optimizer=None,
                        optimizer_idx=None,
                        optimizer_closure=None,
                        on_tpu=None,
                        using_native_amp=None,
                        using_lbfgs=None):        
        # 120 steps ~ 1 epoch
        if self.trainer.global_step < 1000:
            lr_scale = min(1., float(self.trainer.global_step + 1) / 1000.)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * 1e-3

        # update params
        optimizer.step()
        optimizer.zero_grad()

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_simsiam.parameters(), lr=twins_lr,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


# select models
if args.model == 'simclr':
    Model = SimCLRModel
elif args.model == 'simsiam':
    Model = SimSiamModel
elif args.model == 'twins':
    Model = BarlowTwinsModel
elif args.model == 'moco':
    Model = MocoModel
else:
    print('Unsupported model!')

# loop through configurations and train models
gpu_memory_usage = []
runs = []
for seed in range(n_runs):
    pl.seed_everything(seed, workers=True)
    dataloader_train_ssl, dataloader_train_kNN, dataloader_test = get_data_loaders(batch_size)
    model = Model(dataloader_train_kNN)
    trainer = pl.Trainer(max_epochs=max_epochs, gpus=gpus,
                        progress_bar_refresh_rate=100,
                        default_root_dir=logs_root_dir,
                        deterministic=True)
    trainer.fit(
        model,
        train_dataloader=dataloader_train_ssl,
        val_dataloaders=dataloader_test
    )
    gpu_memory_usage.append(torch.cuda.max_memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    runs.append(model.max_accuracy)

    # delete model and trainer + free up cuda memory
    del model
    del trainer
    torch.cuda.empty_cache()


result = np.asarray(runs)
mean = result.mean()
std = result.std()
gpu_usage=np.asarray(gpu_memory_usage).mean()
model = args.model + '_epoch' + args.epoch + '_batch' + args.batch + '_run' + args.runs + '_augs' + args.augs

print(f'{model}: {100*mean:.2f} +- {100*std:.2f}%, GPU used: {gpu_usage / (1024.0**3):.1f} GByte', flush=True)
