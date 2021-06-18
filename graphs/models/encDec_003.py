import torch
from torch import nn
import torch.nn.functional as F
from .base import BaseTemplate


class EncDec_003(BaseTemplate):
    def __init__(self, in_channels=4, features_channels=32):
        super(EncDec_003, self).__init__()

        ##Encoder
        self.in_c = nn.Conv2d(in_channels, 32, kernel_size=1)

        self.conv_enc_1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.GroupNorm(1,32),
            nn.LeakyReLU())

        self.conv_enc_2 =nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.GroupNorm(1, 32),
            nn.LeakyReLU())

        ##Decoder
        self.conv_dec_1 = nn.Conv2d(32, 4, kernel_size=1)

    def encode(self, flows):
        # flows shape BxCxTxHxW
        # B: batch
        # C: Channels
        # T: time (frame)
        # H: Frame height
        # W: Frame width

        out = self.in_c(flows)
        out = out + self.conv_enc_1(out)
        out = out + self.conv_enc_2(out)

        return out

    def decode(self, features):

        out = self.conv_dec_1(features)

        return out

    def forward(self, x):
        # flows shape BxCxTxHxW
        # B: batch
        # C: Channels
        # T: time (frame)
        # H: Frame height
        # W: Frame width

        features = self.encode(x)
        out_flows = self.decode(features)

        return out_flows