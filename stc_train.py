# -*- coding: utf-8 -*-

"""
***

Author: Zhou Ya'nan
Date: 2021-09-16
"""
import time
import argparse
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from stc_classifier import STCClassifier, STCModel
from dataloader import STCData


def parse_args():
    parser = argparse.ArgumentParser(description='Time-series classification using TCN')
    parser.add_argument('--cuda', action='store_false', help='use CUDA (default: True)')
    parser.add_argument('--gpu_ids', default=(0, 1, 2, 3), type=int, nargs=4, help='resolution of the output image.')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers for data loading (default: 2)')

    parser.add_argument('--input_channel', type=int, default=4, help='number of channels of input (default: 4)')
    parser.add_argument('--num_classes', type=int, default=13, help='number of classes in input (default: 13)')
    parser.add_argument('--num_timestamps', type=int, default=48, help='number of timestamps in input (default: 48)')

    parser.add_argument('--ksize', type=int, default=5, help='kernel size (default: 5)')
    parser.add_argument('--levels', type=int, default=2, help='# of levels (default: 4)')
    parser.add_argument('--nhid', type=int, default=120, help='number of hidden units per layer (default: 48)')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout applied to layers (default: 0.05)')

    parser.add_argument('--batch_size', type=int, default=64, help='batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=200, help='upper epoch limit (default: 200)')

    parser.add_argument('--optimizer', type=str, default='adamax', help='optimizer to use (default: adam)')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate (default: 1e-4)')
    parser.add_argument('--loss', type=str, default='ce', help='loss function to use (default: ce)')
    parser.add_argument('--clip_max_norm', default=None, help='gradient clip, None means no clip (default: None)')

    parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='report interval (default: 100')
    args = parser.parse_args()

    return args


def main():
    print("### PyTorch Time-series Classification Training ##################################")

    #################################################################
    # Reading option
    args = parse_args()
    args_cuda = True
    args_gpu_ids = args.gpu_ids
    args_num_workers = args.num_workers

    args_num_classes = args.num_classes
    args_input_channels = args.input_channel
    args_num_timestamps = args.num_timestamps

    args_ksize = args.ksize
    args_levels = args.levels
    args_nhid = args.nhid
    args_dropout = args.dropout

    args_batch_size = args.batch_size
    args_epochs = args.epochs
    args_optimizer = args.optimizer
    args_lr = args.lr
    args_loss = args.loss

    args_loss = 'ce'
    dataset = 'dijon_8m_4mean'

    #################################################################
    # Declaration of data loader
    stc_data = STCData(dataset, batch_size=args_batch_size, num_workers=args_num_workers)

    #################################################################
    # Declaration of network
    stc_classifier = STCClassifier(args_input_channels, args_num_timestamps, args_num_classes)
    model = STCModel(stc_classifier, optimizer=args_optimizer, lr=args_lr, loss=args_loss)

    #################################################################
    # Declaration of trainer
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoint', filename='cp-{epoch:02d}-{val_loss:.2f}', monitor='val_loss')
    early_stop_callback = EarlyStopping(monitor='val_loss')

    trainer = Trainer(accelerator="gpu", max_epochs=args_epochs, gradient_clip_val=None, callbacks=[checkpoint_callback, early_stop_callback])
    trainer.fit(model, stc_data)

    #################################################################
    print('')


if __name__ == "__main__":
    main()
