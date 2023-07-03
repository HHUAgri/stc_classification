# -*- coding: utf-8 -*-

"""
***

Author: Zhou Ya'nan
Date: 2021-09-16
"""

import torch
import torch.nn as nn

from .st_attention import ChannelSpatialSELayer


class SpatialEmbedding(nn.Module):
    def __init__(self, n_inputs, n_outputs, dropout=0.2):
        super(SpatialEmbedding, self).__init__()

        temp_channel = n_outputs//4
        self.embedding_layer = nn.Sequential(
            nn.Conv2d(n_inputs, temp_channel, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(temp_channel),
            nn.ReLU(inplace=True),

            nn.Conv2d(temp_channel, n_outputs, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(n_outputs),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        out = self.embedding_layer(x)
        return out


class SpatialBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, dilation, dropout=0.1):
        super(SpatialBlock, self).__init__()

        temp_channel = 256
        pad_list = [0, dilation]
        self.conv1 = nn.Conv2d(n_inputs, temp_channel, kernel_size=1, stride=1, padding=pad_list[0], dilation=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(temp_channel, n_outputs, kernel_size=3, stride=1, padding=pad_list[1], dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = self.dropout(out)
        return out


class DenseSpatialBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, dropout=0.1):
        super(DenseSpatialBlock, self).__init__()

        out_channels = [n_inputs//4, n_inputs//4, n_inputs//4, n_outputs]
        in_channels = [n_inputs, n_inputs+out_channels[0], n_inputs+out_channels[0]+out_channels[1],
                       n_inputs+out_channels[0]+out_channels[1]+out_channels[2]]

        self.sb1 = SpatialBlock(in_channels[0], n_outputs=out_channels[0], dilation=1, dropout=dropout)
        self.sb2 = SpatialBlock(in_channels[1], n_outputs=out_channels[1], dilation=2, dropout=dropout)
        self.sb3 = SpatialBlock(in_channels[2], n_outputs=out_channels[2], dilation=3, dropout=dropout)
        self.conv4 = nn.Conv2d(in_channels[3], out_channels[3], kernel_size=1, stride=1, padding=0, dilation=1)

    def forward(self, x):
        z1 = self.sb1(x)

        z2 = torch.cat((x, z1), dim=1)
        z2 = self.sb2(z2)

        z3 = torch.cat((x, z1, z2), dim=1)
        z3 = self.sb3(z3)

        z4 = torch.cat((x, z1, z2, z3), dim=1)
        z4 = self.conv4(z4)
        return z4


class VectorBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, padding=0, dropout=0.1):
        super(VectorBlock, self).__init__()

        self.cbrm_layer = nn.Sequential(
            nn.Conv2d(n_inputs, n_inputs, kernel_size=3, stride=1, padding=padding, dilation=1),
            nn.BatchNorm2d(n_inputs),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=padding)
        )

    def forward(self, x):
        out = self.cbrm_layer(x)
        out = self.cbrm_layer(out)
        out = self.cbrm_layer(out)
        out = self.cbrm_layer(out)
        out = torch.flatten(out, start_dim=2, end_dim=-1)
        return out


class DenseSpatialNet(nn.Module):
    def __init__(self, n_inputs=4, n_outputs=256, dropout=0.1):
        super(DenseSpatialNet, self).__init__()

        self.embedding_layer = SpatialEmbedding(n_inputs, n_outputs, dropout=dropout)
        self.dense_layer1 = DenseSpatialBlock(n_outputs, n_outputs, dropout=dropout)
        # self.dense_layer2 = DenseSpatialBlock(n_outputs, n_outputs, dropout=dropout)
        self.attention_layer = ChannelSpatialSELayer(n_outputs)
        self.vector_layer = VectorBlock(n_outputs, n_outputs, padding=1, dropout=dropout)

    def forward(self, x):
        out = self.embedding_layer(x)
        out = self.dense_layer1(out)
        # out = self.dense_layer2(out)
        out = self.attention_layer(out)
        out = self.vector_layer(out)
        return out


def check_parameters(net):
    """
        Returns module parameters. Mb
    """
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10**6


def test_dsb():
    # batch_size x time_step x channels x h x w
    x = torch.randn(9, 12, 4, 32, 32)
    x = torch.reshape(x, [9*12, 4, 32, 32])
    tcn_net = DenseSpatialNet(4, 256)
    print(tcn_net)

    # batch_size x time_step x channels x hw
    y = tcn_net(x)
    y = torch.reshape(y, [9, 12, 256, -1])
    print(str(check_parameters(tcn_net))+' Mb')

    print(y[1][1].shape)


if __name__ == "__main__":
    test_dsb()
