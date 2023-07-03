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
from collections import Counter

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
        def code_to_feat_fillna(label, id, code):

            # prepare path
            label_folder = r'I:\FF\application_dataset\2020-france-agri-grid\s2_l2a_tif_masked\slice_label_32_1_20'
            parcel_folder = r'I:\FF\application_dataset\2020-france-agri-grid\s2_l2a_tif_masked\slice_parcel_32'
            label_parcel_folder = label_folder if (id == 0) else parcel_folder
            feat_folder1 = r'I:\FF\application_dataset\2020-france-agri-grid\s2_l2a_tif_masked\grid_10m_32'
            feat_folder2 = r'I:\FF\application_dataset\2020-france-agri-grid\s2_l2a_tif_masked\grid_20m_10m_32'

            label_parcel_file = os.path.join(label_parcel_folder, '{}.npy'.format(code))
            feat_file1 = os.path.join(feat_folder1, '{}.npy'.format(code[12:]))
            feat_file2 = os.path.join(feat_folder2, '{}.npy'.format(code[12:]))

            # load data
            label_parcel_array = load_numpy_array(label_parcel_file)
            feat_array1 = load_numpy_array(feat_file1)
            feat_array2 = load_numpy_array(feat_file2)

            # combine data
            b1, b2 = 4, 6
            cxt1, h, w = feat_array1.shape
            t = cxt1 // b1

            feat_array1 = feat_array1.reshape([-1, b1, h, w])
            feat_array2 = feat_array2.reshape([-1, b2, h, w])
            feat_array = np.concatenate([feat_array1, feat_array2], axis=1)
            feat_array = feat_array.reshape([-1, h, w])

            # mask non-target values
            label_parcel_mask = (label_parcel_array != label) if (id == 0) else (label_parcel_array != id)
            feat_mask = label_parcel_mask[np.newaxis, :]
            feat_mask = np.repeat(feat_mask, repeats=feat_array.shape[0], axis=0)
            feat_array[feat_mask] = np.nan

            # filling nan with randomly selected pixels from non-nan pixels
            def fillna_random_nonan_3d(np_array):
                nan_pos = np.isnan(np_array[0])
                nz_i, nz_j = np.nonzero(~nan_pos)
                if (nan_pos.sum() > 0) and (nan_pos.sum() < nan_pos.size):
                    choice_pos = np.random.choice(len(nz_i), nan_pos.sum(), replace=True)
                    np_array[:, nan_pos] = np_array[:, nz_i[choice_pos], nz_j[choice_pos]]
            # def
            for i in range(t):
                fillna_random_nonan_3d(feat_array[i * (b1 + b2):(i + 1) * (b1 + b2), :, :])
            # for

            # return
            return feat_array

        def code_to_feat_src(label, id, code):
            """
            根据GRID的CODE，读取对应的特征文件；并根据label标签或parcel标签，对特征数组进行掩膜。
            :param label: GRID的类型
            :param id: GRID的parcel标识
            :param code: GRID的编码
            :return: GRID的时序特征
            """
            # prepare path
            label_folder = r'I:\FF\application_dataset\2020-france-agri-grid\s2_l2a_tif_masked_mvc2\slice_label_32_1_40'
            parcel_folder = r'I:\FF\application_dataset\2020-france-agri-grid\parcel_slice\slice_parcel_32_1'
            feat_folder_1 = r'I:\FF\application_dataset\2020-france-agri-grid\s2_l2a_tif_masked_mvc2\grid_10m_32'
            feat_folder_2 = r'I:\FF\application_dataset\2020-france-agri-grid\s2_l2a_tif_masked_mvc2\grid_20m_10m_32'

            # load grid label or parcel.
            lop_folder = label_folder if (id == 0) else parcel_folder
            lop_file = os.path.join(lop_folder, '{}.npy'.format(code))
            lop_array = load_numpy_array(lop_file)
            lop_mask = (lop_array != label) if (id == 0) else (lop_array != id)

            # load feature data, and combine them.
            feat_file_1 = os.path.join(feat_folder_1, '{}.npy'.format(code[12:]))
            feat_file_2 = os.path.join(feat_folder_2, '{}.npy'.format(code[12:]))
            feat_array_1 = load_numpy_array(feat_file_1)
            feat_array_2 = load_numpy_array(feat_file_2)
            # combine data
            _, h, w = feat_array_1.shape
            feat_array_1 = feat_array_1.reshape([-1, 4, h, w])
            feat_array_2 = feat_array_2.reshape([-1, 6, h, w])
            feat_array = np.concatenate([feat_array_1, feat_array_2], axis=1)
            feat_array = feat_array.reshape([-1, h, w])

            # mask non-target values
            feat_mask = lop_mask[np.newaxis, :]
            feat_mask = np.repeat(feat_mask, repeats=feat_array.shape[0], axis=0)
            feat_array[feat_mask] = np.nan

            return feat_array

        # Fetch the meta infos for samples
        label = self.label_array[item]
        id = self.id_array[item]
        code = self.code_array[item]

        # Get the feature data for samples
        # feat = code_to_feat_fillna(label, id, code)
        feat = code_to_feat_src(label, id, code)
        if self.transform:
            feat = self.transform(feat)

        # [timestamp*channel, height, width] -> [timestamp, channel, height, width]
        cxt, h, w = feat.shape
        feat = feat.reshape([-1, 10, h, w])
        feat[np.isnan(feat)] = 0
        feat = feat / 1000.0

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
        def prepare_meta_data(in_path):
            """

            :param in_path:
            :return:
            """

            def load_metadata(file_path):
                """
                将每个文件名，切分为三部分，并组织到pandas的DataFrame.
                :param file_path:
                :return:
                """
                csv_df = pd.read_csv(file_path, sep=',', header=None)
                csv_lines = list(np.array(csv_df)[:, 0])
                meta_df = pd.DataFrame(index=np.arange(0, len(csv_lines)), columns=['label', 'id', 'code'])
                for ll, line in enumerate(csv_lines):
                    label, id = int(line[0:2]), int(line[3:11])
                    meta_df.at[ll, 'label'] = label
                    meta_df.at[ll, 'id'] = id
                    meta_df.at[ll, 'code'] = line
                return meta_df

            input_df = load_metadata(in_path)
            input_array = np.array(input_df)

            input_label = input_array[:, 0]
            input_id = input_array[:, 1].astype(np.int)
            input_code = input_array[:, 2].astype(np.str)

            return input_label, input_id, input_code

        # CSV file
        train_file = os.path.join('./datasets', self.dataset, "train.csv")
        test_file = os.path.join('./datasets', self.dataset, "test.csv")
        assert(os.path.exists(train_file) or os.path.exists(test_file))

        # load meta data.
        self.train_label, self.train_id, self.train_code = prepare_meta_data(train_file)
        self.test_label, self.test_id, self.test_code = prepare_meta_data(test_file)
        # sample weight
        label_hist = Counter(self.train_label)
        label_hist = sorted(label_hist.items(), key=lambda d: d[0])
        label_hist = np.array(label_hist)[:, 1]
        class_weight = 1 - label_hist / np.sum(label_hist)
        print("### Class Weight: ", class_weight)

        # Move the labels to {0, ..., L-1}
        # labels = np.unique(train_label)
        # transform = {k: i for i, k in enumerate(labels)}
        # train_label = np.vectorize(transform.get)(train_label)
        # test_label = np.vectorize(transform.get)(test_label)

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
