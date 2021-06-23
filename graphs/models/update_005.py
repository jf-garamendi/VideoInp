import torch
from torch import nn
import torch.nn.functional as F
from .base import BaseTemplate
from .custom_layers.partialconv2d import PartialConv2d
from utils.frame_utils import warp


class Update_005(BaseTemplate):
    def __init__(self, in_channels=32 * 3, update='pow', device=None, n_channels_a=64, n_channels_b=32):
        super(Update_005, self).__init__()

        self.device = device

        #down_chs = [in_channels*(2**i) for i in range(0,4)]
        down_chs = [96, 192]
        up_chs = down_chs[::-1]

        self.goDown = Down(down_chs)
        self.goUp = Up(up_chs)
        self.head = nn.Conv2d(up_chs[-1], 32, 3, 1, 1)

        nn.init.xavier_uniform_(self.head.weight)


    def forward(self, x):
        # for a video
        features_in, flow_in, confidence_in = x

        N, C, H, W = flow_in.shape

        new_features = features_in * 0
        confidence_new = confidence_in.clone()

        for n_frame in range(N):
            three_Frames_features, confidence = self.__concat3frames(features_in,
                                                                     flow_for_warping=flow_in,
                                                                     confidence=confidence_in,
                                                                     n_frame=n_frame)

            ftrs, pool_indices  = self.goDown(three_Frames_features, confidence)
            out = self.goUp(ftrs, pool_indices)
            out = self.head(out)


            #new_F = (three_Frames_features[:, 32:64] * confidence_in[n_frame] + out * (1 - confidence_in[n_frame]))
            new_F = out

            new_features[n_frame] = torch.squeeze(new_F)


        return new_features

    def __concat3frames(self, features, flow_for_warping=None, confidence=None, n_frame=1):
        # flows shape BxTxCxHxW
        # B: batch
        # C: Channels
        # T: time (frame)
        # H: Frame height
        # W: Frame width

        N, C, H, W = flow_for_warping.shape

        new_features = features * 0
        new_confidence = confidence.clone()

        frame_flow = flow_for_warping[n_frame]

        ## warping
        if n_frame + 1 < N:
            F_f = warp(features[n_frame + 1, :, :, :], frame_flow[:2, :, :], self.device)
            confidence_f = warp(confidence[n_frame + 1, :, :, :], frame_flow[:2, :, :], self.device)
        else:
            F_f = 0. * features[n_frame]
            confidence_f = 0. * confidence[n_frame]

        if n_frame - 1 >= 0:
            F_b = warp(features[n_frame - 1, :, :, :], frame_flow[2:], self.device)
            confidence_b = warp(confidence[n_frame - 1, :, :, :], frame_flow[2:], self.device)
        else:
            F_b = 0. * features[n_frame]
            confidence_b = 0. * confidence[n_frame]
        # End warping

        # input of the update network is the concatenation of the obtained features from this frame and the neighboring ones
        x = torch.cat((F_b, features[n_frame], F_f), dim=0)

        confidence_in = torch.cat(((confidence_b).repeat(F_b.shape[0], 1, 1),
                                   confidence[n_frame].repeat(features[n_frame].shape[0], 1, 1),
                                   (confidence_f).repeat(F_f.shape[0], 1, 1)),
                                  dim=0)  # same goes for the input mask

        return torch.unsqueeze(x, 0), torch.unsqueeze(confidence_in, 0)


class PartialBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pconv1 = PartialConv2d(multi_channel=True, return_mask=True, kernel_size=(3, 3), padding=1,
                                             in_channels=in_ch, out_channels=out_ch)
        nn.init.xavier_uniform_(self.pconv1.weight)
        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()

    def forward(self, x, mask):
        out, new_confidence = self.pconv1(x, mask)
        out = self.lrelu(out)
        new_confidence = self.relu(new_confidence)

        return out, new_confidence


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class Down(nn.Module):
    def __init__(self, chs=(96, 184, 368, 736)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([PartialBlock(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(2, return_indices=True)

    def forward(self, x, mask):

        indices_list = []
        for block in self.enc_blocks:
            x, mask = block(x, mask)
            x, indices = self.pool(x)
            indices_list.append(indices)

            flattened_mask = mask.flatten(start_dim=2)
            mask = flattened_mask.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)

        return x, indices_list


class Up(nn.Module):
    def __init__(self, chs=(736, 368, 184, 96)):
        super().__init__()
        self.chs = chs
        self.unpool = nn.ModuleList([nn.MaxUnpool2d(2, stride=2) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])

    def forward(self, x, pool_indices):
        unpool_indices = pool_indices[::-1]
        for i in range(len(self.chs) - 1):
            x = self.unpool[i](x, unpool_indices[i])
            x = self.dec_blocks[i](x)
        return x
