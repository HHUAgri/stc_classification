# -*- coding: utf-8 -*-

"""
***

Author: Zhou Ya'nan
Date: 2021-09-16
"""
import time, os, sys
import argparse
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule

from transform_albu import TransformAlb
from data_utils import load_numpy_array


class STCDataset(Dataset):

    def __init__(self, label_array, id_array, file_array, transform=None):
        super(STCDataset, self).__init__()
        self.label_array = label_array
        self.id_array = id_array
        self.file_array = file_array
        self.transform = transform

    def __len__(self):
        return self.file_array.shape[0]

    def __getitem__(self, item):

        label = self.label_array[item]
        id = self.id_array[item]

        file = self.file_array[item]
        feat = load_numpy_array(file).astype(np.float32)
        if self.transform:
            feat = self.transform(feat)

        return {
            "label": torch.as_tensor(label).long(),
            "id": torch.as_tensor(id).long(),
            "feat": feat
        }


class STCData(LightningDataModule):
    def __init__(self, dataset, transform=TransformAlb(), batch_size=16, num_workers=0):
        super().__init__()

        self.dataset = dataset
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_label, self.train_id, self.train_file = None, None, None
        self.test_label, self.test_id, self.test_file = None, None, None
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def prepare_data(self):
        """
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        :return:
        """
        # reading csv file
        train_file = os.path.join('../datasets', self.dataset, "train.csv")
        test_file = os.path.join('../datasets', self.dataset, "test.csv")

        def expend_metadata(file_path):
            csv_df = pd.read_csv(file_path, sep=',', header=None)
            csv_list = list(np.array(csv_df)[:, 0])
            meta_df = pd.DataFrame(columns=['label', 'id', 'path'])
            for ss in csv_list:
                (dir, name_ext) = os.path.split(ss)
                (name, ext) = os.path.splitext(name_ext)
                meta_list = name.split('_')
                label, id = int(meta_list[0]), int(meta_list[1])
                meta_df.append([label, id, ss])
            return meta_df

        train_df = expend_metadata(train_file)
        test_df = expend_metadata(test_file)
        train_array = np.array(train_df)
        test_array = np.array(test_df)

        # Move the labels to {0, ..., L-1}
        labels = np.unique(train_array[:, 0])
        transform = {k: i for i, k in enumerate(labels)}
        train_label = np.vectorize(transform.get)(train_array[:, 0])
        test_label = np.vectorize(transform.get)(test_array[:, 0])

        # deal train and test arrays
        train_id = train_array[:, 1].astype(np.int)
        test_id = test_array[:, 1].astype(np.int)
        train_file = train_array[:, 2].astype(np.str)
        test_file = test_array[:, 2].astype(np.str)

        # warrper data
        self.train_label = train_label
        self.train_id = train_id
        self.train_file = train_file
        self.test_label = test_label
        self.test_id = test_id
        self.test_file = test_file

    def setup(self, stage=None):
        """
        # make assignments here (val/train/test split)
        # called on every process in DDP
        """
        if stage == "fit" or stage is None:
            dastset = STCDataset(self.train_label, self.train_id, self.train_file, self.transform)
            train_size = int(0.8 * len(dastset))
            val_size = len(dastset) - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(dastset, [train_size, val_size])
        if stage == "test" or stage is None:
            self.test_dataset = STCDataset(self.test_label, self.test_id, self.test_file, self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
