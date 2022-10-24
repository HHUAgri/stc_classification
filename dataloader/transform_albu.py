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
    def __init__(self):
        self.aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(always_apply=False, p=0.5),
            # A.ShiftScaleRotate(p=0.5),
            # A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, interpolation=1, border_mode=4, value=None,
            #                    mask_value=None, always_apply=False, approximate=False, p=0.5)
        ])

    def __call__(self, feat):
        augmented = self.aug(feat=feat)
        augmented_feat = augmented['feat']
        return augmented_feat
