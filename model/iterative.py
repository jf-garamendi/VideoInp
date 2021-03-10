from torch import nn
import torch
from .partialconv2d import PartialConv2d

import torch.nn.functional as F


class Flow2features(nn.Module):
    def __init__(self, in_channels=4):
        super(Flow2features, self).__init__()

        self.in_c = nn.Conv2d(in_channels, 32, kernel_size=1)

        self.res1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU())

        self.res2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU())

    def forward(self, x):
        out = self.in_c(x)
        out = out + self.res1(out)
        out = out + self.res2(out)
        return out

class Features2flow(nn.Module):
    def __init__(self, in_channels=32):
        super(Features2flow, self).__init__()

        self.in_c = nn.Conv2d(in_channels, 4, kernel_size=1)

    def forward(self, x):
        out = self.in_c(x)
        return out




#Architecture number 3

class Res_Update(nn.Module):
    def __init__(self, in_channels=32 * 3, update='pow'):
        super(Res_Update, self).__init__()
        self.pconv1 = PartialConv2d(in_channels, 64, multi_channel=True, return_mask=True, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(64)
        self.pconv2 = PartialConv2d(64, 32, multi_channel=True, return_mask=True, kernel_size=(3, 3), padding=1)

    def forward(self, x, mask=None):
        out1, new_mask = self.pconv1(F.leaky_relu(self.bn1(x)), mask)
        out2, new_mask = self.pconv2(F.leaky_relu(self.bn2(out1)), new_mask)

        new_F = (x[:, 32:64] * mask[:, 32:33] + out2 * (1 - mask[:, 32:33]))

        return new_F, new_mask[:,0]

