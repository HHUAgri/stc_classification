# -*- coding: utf-8 -*-

"""
***

Author: Zhou Ya'nan
Date: 2021-09-16
"""

import torch
import torch.nn as nn

from tcn import TemporalConvNet
from dsn import DenseSpatialNet


class SpatialTemporalConvNet(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_timestep, dropout=0.2):
        super(SpatialTemporalConvNet, self).__init__()

        st_channel = 256
        tcn_channels = [64, 32, 32, 16, 8]
        self.dsn_layer = DenseSpatialNet(n_inputs, st_channel)
        self.tcn_layer = TemporalConvNet(st_channel, num_channels=tcn_channels)

        self.dense_layer = nn.Sequential(
            nn.Linear(tcn_channels[-1]*n_timestep, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, n_outputs)
        )

    def forward(self, x):
        [b, t, c, h, w] = x.shape
        x = torch.reshape(x, [b*t, c, h, w])

        out = self.dsn_layer(x)
        out = torch.reshape(out, [b, t, -1])
        out = out.permute([0, 2, 1])

        out = self.tcn_layer(out)
        out = torch.flatten(out, 1, -1)

        out = self.dense_layer(out)
        return out


def check_parameters(net):
    """
        Returns module parameters. Mb
    """
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10**6


def test_stc():
    # batch_size x time_step x channels x h x w
    x = torch.randn(9, 12, 4, 32, 32)
    stcn_net = SpatialTemporalConvNet(4, 8, 12)
    print(stcn_net)

    # class
    y = stcn_net(x)
    print(str(check_parameters(stcn_net))+' Mb')
    print(y)


if __name__ == "__main__":
    test_stc()
