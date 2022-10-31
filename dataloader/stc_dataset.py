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

from .transform_albu import TransformAlb
from .data_utils import load_numpy_array


class STCDataset(Dataset):

    def __init__(self, label_array, id_array, code_array, transform=None):
        super(STCDataset, self).__init__()
        self.label_array = label_array
        self.id_array = id_array
        self.code_array = code_array
        self.transform = transform

    def __len__(self):
        return self.code_array.shape[0]

    def __getitem__(self, item):
        def code_to_feat(label, id, code):
            # prepare path
            label_folder = r'K:\FF\application_dataset\2020-france-agri-grid\s2_l2a_tif_masked\slice_label_32'
            parcel_folder = r'K:\FF\application_dataset\2020-france-agri-grid\s2_l2a_tif_masked\slice_parcel_32'
            label_parcel_folder = label_folder if (id == 0) else parcel_folder
            feat_folder1 = r'K:\FF\application_dataset\2020-france-agri-grid\s2_l2a_tif_masked\grid_10m_32'
            feat_folder2 = r'K:\FF\application_dataset\2020-france-agri-grid\s2_l2a_tif_masked\grid_20m_10m_32'

            label_parcel_file = os.path.join(label_parcel_folder, '{}.npy'.format(code))
            feat_file1 = os.path.join(feat_folder1, '{}.npy'.format(code[12:]))
            feat_file2 = os.path.join(feat_folder2, '{}.npy'.format(code[12:]))

            # load data
            label_parcel_array = load_numpy_array(label_parcel_file)
            feat_array1 = load_numpy_array(feat_file1)
            feat_array2 = load_numpy_array(feat_file2)

            # combine data
            cxt, h, w = feat_array1.shape
            feat_array1 = feat_array1.reshape([-1, 4, h, w])
            feat_array2 = feat_array2.reshape([-1, 6, h, w])
            feat_array = np.concatenate([feat_array1, feat_array2], axis=1)
            feat_array = feat_array.reshape([-1, h, w])

            # mask non-target values
            label_parcel_mask = (label_parcel_array != label) if (id == 0) else (label_parcel_array != id)
            feat_mask = label_parcel_mask[np.newaxis, :]
            feat_mask = np.repeat(feat_mask, repeats=feat_array.shape[0], axis=0)
            feat_array[feat_mask] = 0

            return feat_array

        label = self.label_array[item]
        id = self.id_array[item]
        code = self.code_array[item]

        feat = code_to_feat(label, id, code)
        # if self.transform:
        #     feat = self.transform(feat)

        # [timestamp*channel, height, width] -> [timestamp, channel, height, width]
        cxt, h, w = feat.shape
        feat = feat.reshape([-1, 10, h, w])

        # print(code)
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

        self.train_label, self.train_id, self.train_code = None, None, None
        self.test_label, self.test_id, self.test_code = None, None, None
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def prepare_data(self):
        """
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        :return:
        """
        # reading csv file
        train_file = os.path.join('./datasets', self.dataset, "train.csv")
        test_file = os.path.join('./datasets', self.dataset, "test.csv")

        def expend_metadata(file_path):
            csv_df = pd.read_csv(file_path, sep=',', header=None)
            csv_lines = list(np.array(csv_df)[:, 0])
            meta_df = pd.DataFrame(index=np.arange(0, len(csv_lines)), columns=['label', 'id', 'code'])
            for ll, line in enumerate(csv_lines):
                label, id = int(line[0:2]), int(line[3:11])
                meta_df.at[ll, 'label'] = label
                meta_df.at[ll, 'id'] = id
                meta_df.at[ll, 'code'] = line
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
        train_code = train_array[:, 2].astype(np.str)
        test_code = test_array[:, 2].astype(np.str)

        # warrper data
        self.train_label = train_label
        self.train_id = train_id
        self.train_code = train_code
        self.test_label = test_label
        self.test_id = test_id
        self.test_code = test_code

    def setup(self, stage=None):
        """
        # make assignments here (val/train/test split)
        # called on every process in DDP
        """
        if stage == "fit" or stage is None:
            dastset = STCDataset(self.train_label, self.train_id, self.train_code, self.transform)
            train_size = int(0.8 * len(dastset))
            val_size = len(dastset) - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(dastset, [train_size, val_size])
        if stage == "test" or stage is None:
            self.test_dataset = STCDataset(self.test_label, self.test_id, self.test_code, self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
