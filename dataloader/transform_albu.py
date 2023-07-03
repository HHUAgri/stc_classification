# -*- coding: utf-8 -*-
"""

Author:
Date:
"""
import os
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import torch


class TransformAlb(object):
    def __init__(self, probability=0.2):
        self.aug = A.Compose([
            A.HorizontalFlip(p=probability),
            A.VerticalFlip(p=probability),
            A.Transpose(),
            A.RandomRotate90(always_apply=False, p=probability),
            # A.ShiftScaleRotate(p=probability),
            A.RandomResizedCrop(32, 32, p=probability),
            A.GridDistortion(),
            A.ElasticTransform(),
            A.RandomGridShuffle()
        ])

    def __call__(self, feat):
        feat = feat.transpose((1, 2, 0))
        augmented = self.aug(image=feat)
        augmented_feat = augmented['image']
        return augmented_feat.transpose((2, 0, 1))
