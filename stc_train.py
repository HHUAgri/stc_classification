# -*- coding: utf-8 -*-

"""
***

Author: Zhou Ya'nan
Date: 2021-09-16
"""
import time
import argparse
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from stc_classifier import STCClassifier, STCModel
from dataloader import STCData

seed_everything(42)


def parse_args():
    parser = argparse.ArgumentParser(description='Time-series classification using TCN')
    parser.add_argument('--cuda', action='store_false', help='use CUDA (default: True)')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for data loading (default: 2)')

    parser.add_argument('--input_channel', type=int, default=10, help='number of channels of input (default: 10)')
    parser.add_argument('--num_classes', type=int, default=13, help='number of classes in input (default: 13)')
    parser.add_argument('--num_timestamps', type=int, default=24, help='number of timestamps in input (default: 48)')

    parser.add_argument('--ksize', type=int, default=5, help='kernel size (default: 5)')
    parser.add_argument('--levels', type=int, default=2, help='# of levels (default: 4)')
    parser.add_argument('--nhid', type=int, default=120, help='number of hidden units per layer (default: 48)')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout applied to layers (default: 0.05)')

    parser.add_argument('--batch_size', type=int, default=2, help='batch size (default: 16)')
    parser.add_argument('--epochs', type=int, default=200, help='upper epoch limit (default: 200)')

    parser.add_argument('--optimizer', type=str, default='adamax', help='optimizer to use (default: adamax)')
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
    args_seed = 1027
    args_gpu = args.gpu
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
    dataset = 'dijon_mvc2_1_40'
    class_weight = [0.0001, 0.7563, 0.9161, 0.9218, 0.9916, 0.9577, 0.9603, 0.9687, 0.9802, 0.9935, 0.9832, 0.5748, 0.9958]
    # dataset = 'dijon_mvc2_1_20'
    # class_weight = [0.0001, 0.7486, 0.8978, 0.9210, 0.9910, 0.9482, 0.9495, 0.9587, 0.9753, 0.9979, 0.9770, 0.6424, 0.9925]

    #################################################################
    # Declaration of data loader
    stc_data = STCData(dataset, batch_size=args_batch_size, num_workers=args_num_workers)

    #################################################################
    # Declaration of network
    stc_classifier = STCClassifier(args_input_channels, args_num_timestamps, args_num_classes)
    model = STCModel(stc_classifier, optimizer=args_optimizer, lr=args_lr, loss=args_loss, class_weight=class_weight)

    #################################################################
    # Declaration of trainer
    logger = TensorBoardLogger('stc_logs', name='stc_model')
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoint', filename='cp-{epoch:02d}-{val_loss:.2f}',
                                          monitor='val_loss',
                                          save_last=False, save_top_k=-1)
    early_stop_callback = EarlyStopping(monitor='val_loss')

    trainer = Trainer(
        logger=logger,
        accelerator="gpu", gpus=1,
        max_epochs=args_epochs, min_epochs=10,
        gradient_clip_val=None,
        callbacks=[checkpoint_callback]
    )
    ckpt_path = r'E:\develop_project\github\stc_classification\checkpoint\cp-epoch=57-val_loss=1.02.ckpt'
    trainer.fit(model, stc_data, ckpt_path=None)

    #################################################################
    print('### Training complete !')


if __name__ == "__main__":
    main()
