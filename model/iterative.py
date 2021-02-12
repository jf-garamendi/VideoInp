import torch.nn as nn
import torch
from .partialconv2d import PartialConv2d

import torch.nn.functional as F
class Features2flow(nn.Module):
    def __init__(self, in_channels=32):
        super(Features2flow, self).__init__()

        self.in_c = nn.Sequential(nn.Conv2d(in_channels, 4, kernel_size=1))

    def forward(self, x):
        out = self.in_c(x)
        return out


class Flow2features(nn.Module):
    def __init__(self, in_channels=4):
        super(Flow2features, self).__init__()

        self.in_c = nn.Sequential(nn.Conv2d(in_channels, 32, 1))

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


class Res_Update2(nn.Module):
    def __init__(self, in_channels=32 * 3, update='pow'):
        super(Res_Update2, self).__init__()
        self.initial_mask = Initial_mask()
        self.pconv1 = PartialConv2d(multi_channel='semi', return_mask=True, kernel_size=(3, 3), padding=1,
                                    in_channels=in_channels, out_channels=64, update=update)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(64)
        self.pconv2 = PartialConv2d(multi_channel=False, return_mask=True, kernel_size=(3, 3), padding=1,
                                    in_channels=64, out_channels=32, update=update)
        # self.bn3=nn.BatchNorm2d(64)
        # self.pconv3=PartialConv2d(multi_channel=False,return_mask=True,kernel_size=(3,3),padding=1,in_channels=64,out_channels=32)

    def forward(self, x, mask=None):
        out1, new_mask1 = self.pconv1(F.leaky_relu(self.bn1(x)), mask)

        out2, new_mask2 = self.pconv2(F.leaky_relu(self.bn2(out1)), new_mask1)

        # out3,_=self.pconv3(F.leaky_relu(self.bn3(out2+out1)),new_mask)

        return (x[:, 32:64] + out2 * new_mask2 * (1 - mask[:, 32:33])) / (
                    1 + new_mask2 * (1 - mask[:, 32:33])), new_mask2


class Res_Update3(nn.Module):
    def __init__(self, in_channels=32 * 3, update='pow'):
        super(Res_Update3, self).__init__()
        self.initial_mask = Initial_mask()
        self.pconv1 = PartialConv2d(multi_channel='semi', return_mask=True, kernel_size=(3, 3), padding=1,
                                    in_channels=in_channels, out_channels=64, update=update)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(64)
        self.pconv2 = PartialConv2d(multi_channel=False, return_mask=True, kernel_size=(3, 3), padding=1,
                                    in_channels=64, out_channels=32, update=update)
        # self.bn3=nn.BatchNorm2d(64)
        # self.pconv3=PartialConv2d(multi_channel=False,return_mask=True,kernel_size=(3,3),padding=1,in_channels=64,out_channels=32)

    def forward(self, x, mask=None):
        out1, new_mask = self.pconv1(F.leaky_relu(self.bn1(x)), mask)

        out2, new_mask = self.pconv2(F.leaky_relu(self.bn2(out1)), new_mask)

        # out3,_=self.pconv3(F.leaky_relu(self.bn3(out2+out1)),new_mask)

        return (x[:, 32:64] * mask[:, 32:33] + out2 * (1 - mask[:, 32:33])), new_mask


class Res_Update4(nn.Module):
    def __init__(self, in_channels=32 * 3, update='pow'):
        super(Res_Update4, self).__init__()
        self.pconv1 = PartialConv2d(multi_channel='semi', return_mask=True, kernel_size=(3, 3), padding=1,
                                    in_channels=in_channels, out_channels=64, update=update)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(64)
        self.pconv2 = PartialConv2d(multi_channel=False, return_mask=True, kernel_size=(3, 3), padding=1,
                                    in_channels=64, out_channels=32, update=update)
        # self.bn3=nn.BatchNorm2d(64)
        # self.pconv3=PartialConv2d(multi_channel=False,return_mask=True,kernel_size=(3,3),padding=1,in_channels=64,out_channels=32)

    def forward(self, x, mask=None):
        out1, new_mask = self.pconv1(F.leaky_relu(self.bn1(x)), mask)

        out2, new_mask = self.pconv2(F.leaky_relu(self.bn2(out1)), new_mask)

        # out3,_=self.pconv3(F.leaky_relu(self.bn3(out2+out1)),new_mask)

        return (x[:, 32:64] * mask[:, 32:33] + out2 * (1 - mask[:, 32:33]) * new_mask), new_mask

class Initial_mask(nn.Module):
    def __init__(self):
        super(Initial_mask, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 4, 3, 2, 1, bias=False), nn.BatchNorm2d(4), nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(4, 8, 3, 2, 1, bias=False), nn.BatchNorm2d(8), nn.LeakyReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(8, 16, 3, 2, 1, bias=False), nn.BatchNorm2d(16), nn.LeakyReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(16, 16, 3, 2, 1, bias=False), nn.BatchNorm2d(16), nn.LeakyReLU())

        self.up3 = nn.Sequential(nn.Conv2d(32, 16, 1, bias=False), nn.BatchNorm2d(16), nn.LeakyReLU())
        self.up2 = nn.Sequential(nn.Conv2d(24, 8, 1, bias=False), nn.BatchNorm2d(8), nn.LeakyReLU())
        self.up1 = nn.Sequential(nn.Conv2d(12, 8, 1, bias=False), nn.BatchNorm2d(8), nn.LeakyReLU())
        self.out = nn.Conv2d(8, 1, 1, bias=False)

    def forward(self, x):
        _, _, h0, w0 = x.shape
        x1 = self.conv1(-x + .5)
        _, _, h1, w1 = x1.shape
        x2 = self.conv2(x1)
        _, _, h2, w2 = x2.shape
        x3 = self.conv3(x2)
        _, _, h3, w3 = x3.shape
        x4 = self.conv4(x3)

        y3 = self.up3(torch.cat((F.interpolate(x4, size=(h3, w3), mode='bilinear', align_corners=True), x3), dim=1))
        y2 = self.up2(torch.cat((F.interpolate(y3, size=(h2, w2), mode='bilinear', align_corners=True), x2), dim=1))
        y1 = self.up1(torch.cat((F.interpolate(y2, size=(h1, w1), mode='bilinear', align_corners=True), x1), dim=1))
        out = self.out(F.interpolate(y1, size=(h0, w0), mode='bilinear', align_corners=True))
        return torch.sigmoid(out - 1) * (x == 0) + x * (x == 1)

