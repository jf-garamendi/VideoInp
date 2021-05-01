from torch import nn
import torch
from .partialconv2d import PartialConv2d

import torch.nn.functional as F


class Flow2features(nn.Module):
    def __init__(self, in_channels=4):
        super(Flow2features, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, 1, 1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU()
        )

    def forward(self, x):
        # to CxNframexHxW
        x = torch.unsqueeze(x.permute(1,0,2,3), 0)

        out = x
        out = self.conv1(out)
        out = out + self.conv2(out)

        out = torch.squeeze(out).permute(1,0,2,3)

        return out

class Features2flow(nn.Module):
    def __init__(self, features_channels=32):
        super(Features2flow, self).__init__()

        self.in_c = nn.Conv3d(features_channels, 4, kernel_size=1)

    def forward(self, x):
        x = torch.unsqueeze(x.permute(1,0,2,3), 0)

        out = self.in_c(x)

        out = torch.squeeze(out).permute(1, 0, 2, 3)
        return out




#Architecture number 3

class Res_Update(nn.Module):
    def __init__(self, in_channels=32 * 3, update='pow'):
        super(Res_Update, self).__init__()
        self.pconv1 = PartialConv2d(multi_channel='semi', return_mask=True, kernel_size=(3, 3), padding=1,
                                    in_channels=in_channels, out_channels=64, update=update)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(64)
        self.pconv2 = PartialConv2d(multi_channel=False, return_mask=True, kernel_size=(3, 3), padding=1,
                                    in_channels=64, out_channels=32, update=update)
    def forward(self, x, mask=None):
        x = torch.unsqueeze(x, 0)
        mask = torch.unsqueeze(mask,0 )
        out1, new_mask = self.pconv1(F.leaky_relu(self.bn1(x)), mask)
        out2, new_mask = self.pconv2(F.leaky_relu(self.bn2(out1)), new_mask)

        new_F = (x[:, 32:64] * mask[:, 32:33] + out2 * (1 - mask[:, 32:33]))

        return torch.squeeze(new_F), torch.squeeze(new_mask[:,0])

