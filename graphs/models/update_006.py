import torch
from torch import nn
import torch.nn.functional as F
from .base import BaseTemplate
from .custom_layers.partialconv2d_pierrick import PartialConv2d_pierrick
from utils.frame_utils import warp


class Update_006(BaseTemplate):
    def __init__(self, in_channels=32 * 3, update='pow', device=None, n_channels_a = 64, n_channels_b = 32):
        super(Update_006, self).__init__()

        self.device = device

        self.pconvA_1 = PartialConv2d_pierrick(multi_channel='semi', return_mask=True, kernel_size=(3, 3), padding=1,
                                             in_channels=in_channels, out_channels=in_channels*2, update=update)

        self.pconvA_2 = PartialConv2d_pierrick(multi_channel=False, return_mask=True, kernel_size=(3, 3), padding=1,
                                             in_channels=in_channels*2, out_channels=in_channels*3, update=update)
        self.pconvA_3 = PartialConv2d_pierrick(multi_channel=False, return_mask=True, kernel_size=(3, 3), padding=1,
                                             in_channels=in_channels*3, out_channels=in_channels, update=update)
        self.pconvA_4 = PartialConv2d_pierrick(multi_channel=False, return_mask=True, kernel_size=(3, 3), padding=1,
                                             in_channels=in_channels, out_channels=in_channels, update=update)

        self.pconvB_1 = PartialConv2d_pierrick(multi_channel=False, return_mask=True, kernel_size=(3, 3), padding=1,
                                             in_channels=in_channels, out_channels=in_channels, update=update)

        self.pconvB_2 = PartialConv2d_pierrick(multi_channel=False, return_mask=True, kernel_size=(3, 3), padding=1,
                                             in_channels=in_channels, out_channels=in_channels, update=update)
        self.pconvB_3 = PartialConv2d_pierrick(multi_channel=False, return_mask=True, kernel_size=(3, 3), padding=1,
                                             in_channels=in_channels, out_channels=in_channels, update=update)
        self.pconvB_4 = PartialConv2d_pierrick(multi_channel=False, return_mask=True, kernel_size=(3, 3), padding=1,
                                             in_channels=in_channels, out_channels=in_channels, update=update)

        ##Encoder
        self.in_c = nn.Conv2d(in_channels, 32, kernel_size=1)

        self.conv_enc_1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.GroupNorm(1, 32),
            nn.LeakyReLU())

        self.conv_enc_2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.GroupNorm(1, 32),
            nn.LeakyReLU())

        ##Decoder
        self.conv_dec_1 = nn.Conv2d(32, 32, kernel_size=1)


    def forward(self, x):
        # for a video
        features_in, flow_in, confidence_in = x

        N, C, H, W = flow_in.shape

        new_features = features_in*0
        confidence_new = confidence_in.clone()


        for n_frame in range(N):
            #print(n_frame)
            three_Frames_features, confidence = self.__iterative_step(features_in,
                                                       flow_for_warping=flow_in,
                                                       confidence=confidence_in,
                                                       n_frame=n_frame)



            out, new_confidence = self.pconvA_1(three_Frames_features, confidence)
            out, new_confidence = self.pconvA_2(F.leaky_relu(out), new_confidence)
            out, new_confidence = self.pconvA_3(F.leaky_relu(out), new_confidence)
            out, new_confidence = self.pconvA_4(F.leaky_relu(out), new_confidence)

            out, new_confidence = self.pconvB_1(F.leaky_relu(out), new_confidence)
            out, new_confidence = self.pconvB_2(F.leaky_relu(out), new_confidence)
            out, new_confidence = self.pconvB_3(F.leaky_relu(out), new_confidence)
            out, new_confidence = self.pconvB_4(F.leaky_relu(out), new_confidence)

            out = self.in_c(out)
            out = out + self.conv_enc_1(out)
            out = out + self.conv_enc_2(out)
            out = self.conv_dec_1(out)



            new_F = (three_Frames_features[:, 32:64] * confidence_in[n_frame] + out * (1 - confidence_in[n_frame]))

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