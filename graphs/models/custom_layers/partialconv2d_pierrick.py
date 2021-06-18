###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
###############################################################################

import torch
import torch.nn.functional as F
from torch import nn, cuda
from torch.autograd import Variable


class PartialConv2d_pierrick(nn.Conv2d):
    def __init__(self, update='pow', *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d_pierrick, self).__init__(*args, **kwargs)
        self.update = update
        print(self.update)

        self.a = torch.nn.Parameter(torch.tensor(.33 / 10), requires_grad=True)
        self.b = torch.nn.Parameter(torch.tensor(.666 / 10), requires_grad=True)
        self.p = torch.nn.Parameter(torch.tensor(1. / 10), requires_grad=True)
        if self.multi_channel == True:
            self.weight_maskUpdater = torch.nn.Parameter(
                torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]),
                requires_grad=True)
        elif self.multi_channel == 'semi':
            self.weight_maskUpdater = torch.nn.Parameter(
                torch.ones(1, self.in_channels, self.kernel_size[0], self.kernel_size[1]), requires_grad=True)
        else:
            self.weight_maskUpdater = torch.nn.Parameter(torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1]),
                                                         requires_grad=True)

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                             self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            # with torch.no_grad():
            if True:

                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2],
                                          input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in

                if self.update == 'pow':
                    if self.multi_channel != False:
                        non_border = (input.abs().sum(dim=0).sum(dim=-1).sum(dim=-1) != 0).view(1, -1, 1, 1)

                        p = torch.sign(self.p) * (torch.abs(10 * self.p) + 10 ** -3)
                        self.update_mask = F.conv2d(torch.clamp(mask, 1e-5, 1) ** p,
                                                    non_border * self.weight_maskUpdater.abs(), bias=None,
                                                    stride=self.stride, padding=self.padding, dilation=self.dilation,
                                                    groups=1)
                        self.slide_winsize = (non_border * self.weight_maskUpdater).abs().sum()  # for trainable params
                    else:
                        p = torch.sign(self.p) * (torch.abs(self.p * 10) + 10 ** -3)
                        self.update_mask = F.conv2d(torch.clamp(mask, 1e-5, 1) ** p, self.weight_maskUpdater.abs(),
                                                    bias=None, stride=self.stride, padding=self.padding,
                                                    dilation=self.dilation, groups=1)
                        self.slide_winsize = self.weight_maskUpdater.abs().sum()
                    self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-5)

                    self.update_mask = torch.clamp(
                        (torch.clamp(self.update_mask, 1e-5, None) / self.slide_winsize) ** (1 / p), 1e-5, 1)
                elif self.update == 'pol':
                    self.update_mask = F.conv2d(torch.clamp(mask, 1e-5, 1), self.weight_maskUpdater.abs(), bias=None,
                                                stride=self.stride, padding=self.padding, dilation=self.dilation,
                                                groups=1)
                    self.slide_winsize = self.weight_maskUpdater.abs().sum()  # for trainable params
                    self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-5)
                    mean = self.update_mask / self.slide_winsize
                    a = 10 * self.a
                    b = 10 * self.b
                    self.update_mask = torch.clamp(
                        (27 * a / 2 - 27 * b / 2 + 9 / 2) * mean ** 3 + (-45 * a / 2 + 18 * b - 9 / 2) * mean ** 2 + (
                                    9 * a - 9 * b / 2 + 1) * mean, 0, 1)
                # for mixed precision training, change 1e-8 to 1e-6

                # self.slide_winsize=self.weight_maskUpdater.abs().sum()# for trainable params
                # self.mask_ratio = self.slide_winsize/(self.update_mask + 1e-8)

                # self.update_mask = torch.clamp(self.update_mask, 0, 1)
                # self.update_mask = torch.clamp(self.update_mask/self.slide_winsize*1.5, 0, 1) #conf>.5 => 1

                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super(PartialConv2d_pierrick, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output