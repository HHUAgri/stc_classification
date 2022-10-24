# -*- coding: utf-8 -*-

"""
***

Author: Zhou Ya'nan
Date: 2021-09-16
"""
import torch
from torch import nn
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning import LightningModule

from stc_model import SpatialTemporalConvNet
from stc_loss import STCLosses


class STCClassifier(nn.Module):
    def __init__(self, in_channels, in_timestep, out_classes):
        """

        :param in_channels: 输入数据的通道数(单一时间点)
        :param in_timestep: 输入数据的时间步数
        :param out_classes: 输出分类的类型数
        """
        super(STCClassifier, self).__init__()
        self.net = SpatialTemporalConvNet(n_inputs=in_channels, n_outputs=out_classes, n_timestep=in_timestep)

    def forward(self, inputs):
        out = self.net(inputs)
        return F.log_softmax(out, dim=1)


class STCModel(LightningModule):
    def __init__(self, model,
                 optimizer="adam",
                 lr=0.0001,
                 loss='ce'
                 ):
        super().__init__()

        self.model = model.to(self.device)

        self.optimizer = optimizer
        self.lr = lr

        self.criterion = STCLosses().build_loss(loss)

        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        feat, label = batch["feat"], batch["label"]
        label_pred = self.model(feat)

        loss = self.criterion(label_pred, label)
        self.log("loss", loss, prog_bar=True)

        # Here, can record many metrics
        pred = torch.argmax(label_pred, dim=1)
        acc = self.accuracy(pred, label)
        self.log("acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):

        feat, label = batch["feat"], batch["label"]
        label_pred = self.model(feat)

        loss = self.criterion(label_pred, label)
        self.log("val_loss", loss, prog_bar=True)

        # Here, can record many metrics
        pred = torch.argmax(label_pred, dim=1)
        acc = self.accuracy(pred, label)
        self.log("val_acc", acc, prog_bar=True)
        # return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        label, id, feat = batch["label"], batch["id"], batch["feat"]
        return self(feat)

    def configure_optimizers(self):
        supported_optimizer = {
            "sgd": torch.optim.SGD,  # momentum, weight_decay, lr
            "rmsprop": torch.optim.RMSprop,  # momentum, weight_decay, lr
            "adam": torch.optim.Adam,  # weight_decay, lr
            "adadelta": torch.optim.Adadelta,  # weight_decay, lr
            "adagrad": torch.optim.Adagrad,  # lr, lr_decay, weight_decay
            "adamax": torch.optim.Adamax  # lr, weight_decay
        }
        if self.optimizer not in supported_optimizer:
            raise ValueError("Now only support optimizer {}".format(self.optimizer))

        optimizer_kwargs = {'lr': self.lr, 'weight_decay': 1e-5}
        optimizer = supported_optimizer[self.optimizer](self.model.parameters(), **optimizer_kwargs)

        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10),
            'monitor': 'val_loss'}

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
