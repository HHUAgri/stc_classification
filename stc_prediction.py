# -*- coding: utf-8 -*-

"""
***

Author: Zhou Ya'nan
Date: 2021-09-16
"""
import time
import argparse
from pytorch_lightning import Trainer

from stc_classifier import STCModel
from dataloader import STCData


def parse_args():
    parser = argparse.ArgumentParser(description='Time-series classification using TCN')
    # parser.add_argument('--cuda', action='store_false', help='use CUDA (default: True)')
    # parser.add_argument('--gpu_ids', default=(0, 1, 2, 3), type=int, nargs=4, help='resolution of the output image.')
    parser.add_argument('--dataset', type=str, default='dijon', help='dataset to use')
    parser.add_argument('--cp_path', type=str, default='cp.ckpt', help='checkpoint to load')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size (default: 64)')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers for data loading (default: 2)')
    args = parser.parse_args()
    return args


def main():
    print("### PyTorch Time-series Classification Training ##################################")

    #################################################################
    # Reading option
    args = parse_args()
    args_dataset = args.dataset
    args_checkpoint_path = args.checkpoint_path
    args_batch_size = args.batch_size
    args_num_workers = args.num_workers

    args_dataset = 'dijon_8m_4mean'
    checkpoint_path = r''

    # Declaration of data loader
    stc_data = STCData(args_dataset, batch_size=args_batch_size, num_workers=args_num_workers)

    # Loading network
    trained_model = STCModel.load_from_checkpoint(checkpoint_path)

    # Declaration of trainer
    trainer = Trainer(accelerator="gpu")
    pred_label = trainer.predict(trained_model, stc_data)

    #################################################################
    print('')


if __name__ == "__main__":
    main()
