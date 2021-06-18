import torch
from torch import nn
import torch.nn.functional as F
from .base import BaseTemplate
from .custom_layers.partialconv2d_pierrick import PartialConv2d_pierrick
from utils.frame_utils import warp


class Update_003(BaseTemplate):
    def __init__(self, in_channels=32 * 3, update='pow', device=None):
        super(Update_003, self).__init__()

        self.device = device

        self.pconv1 = PartialConv2d_pierrick(multi_channel='semi', return_mask=True, kernel_size=(3, 3), padding=1,
                                             in_channels=in_channels, out_channels=64, update=update)
        self.ln1 = nn.GroupNorm(1,in_channels)
        self.ln2 = nn.GroupNorm(1,64)
        self.pconv2 = PartialConv2d_pierrick(multi_channel=False, return_mask=True, kernel_size=(3, 3), padding=1,
                                             in_channels=64, out_channels=32, update=update)

    def forward(self, x):
        # for a video
        features_in, flow_in, confidence_in = x

        N, C, H, W = flow_in.shape

        new_features = features_in*0
        confidence_new = confidence_in.clone()


        for n_frame in range(N):
            x_, confidence = self.__iterative_step(features_in,
                                                       flow_for_warping=flow_in,
                                                       confidence=confidence_in,
                                                       n_frame=n_frame)



            out1, new_confidence = self.pconv1(F.leaky_relu(self.ln1(x_)), confidence)
            out2, new_confidence = self.pconv2(F.leaky_relu(self.ln2(out1)), new_confidence)

            new_F = (x_[:, 32:64] * confidence[:, 32:33] + out2 * (1 - confidence[:, 32:33]))

            new_features[n_frame] = torch.squeeze(new_F)
            confidence_new[n_frame] = torch.squeeze(new_confidence)

            # force the initially confident pixels to stay confident, because a decay can be observed
            # depending on the update rule of the partial convolution
            confidence_new[n_frame][confidence_in[n_frame] == 1] = 1.

        return new_features, confidence_new

    def __iterative_step(self, features, flow_for_warping=None, confidence=None, n_frame=1):
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

