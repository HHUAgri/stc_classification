# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction=self.reduction)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class STCLosses(object):

    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """
        Choices: ['bce' or 'nll']
        """
        if mode == 'ce':
            return F.cross_entropy
        elif mode == 'nll':
            return F.nll_loss
        elif mode == 'focal':
            return self.focal_loss()
        else:
            raise NotImplementedError

    def focal_loss(self, gamma=2.0, reduction='mean'):
        """Factory function for FocalLoss.
        Args:
            alpha (Sequence, optional): Weights for each class. Will be converted
                to a Tensor if not None. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
            device (str, optional): Device to move alpha to. Defaults to 'cpu'.
            dtype (torch.dtype, optional): dtype to cast alpha to.
                Defaults to torch.float32.
        Returns:
            A FocalLoss object
        """
        return FocalLoss(gamma=gamma, reduction=reduction)


if __name__ == "__main__":

    pred = torch.randn((3, 5))
    label = torch.tensor([2, 3, 4])

    loss_fun = STCLosses().build_loss('focal')(pred, label)
    print(loss_fun)
