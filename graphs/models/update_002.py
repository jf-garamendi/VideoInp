import torch
from torch import nn
import torch.nn.functional as F
from .base import BaseTemplate
from .custom_layers.partialconv2d import PartialConv2d
from utils.frame_utils import warp


class Update_002(BaseTemplate):
    def __init__(self, in_channels=32 * 3, update='pow', device=None):
        super(Update_002, self).__init__()

        self.device = device

        self.pconv1 = PartialConv2d(multi_channel=True, return_mask=True, kernel_size=(3, 3), padding=1,
                                             in_channels=in_channels, out_channels=64)
        self.pconv2 = PartialConv2d(multi_channel=True, return_mask=True, kernel_size=(3, 3), padding=1,
                                             in_channels=64, out_channels=32)

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



            out1, new_confidence = self.pconv1(F.leaky_relu(x_), confidence)
            out2, new_confidence = self.pconv2(F.leaky_relu(out1), new_confidence)

            new_F = (x_[:, 32:64] * confidence[:, 32:33] + out2 * (1 - confidence[:, 32:33]))

            new_features[n_frame] = torch.squeeze(new_F)
            confidence_new[n_frame] = torch.squeeze(new_confidence[0,1])

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

    '''
    def training_one_batch(self, batch):
        return self.__full_iterative_scheme(batch, training=True)

    def validating_one_batch(self, batch):
        return self.__full_iterative_scheme(batch, training=False)

    def inferring_one_batch(self, batch):
        return self.__full_iterative_scheme(batch, training=False)



    def __full_iterative_scheme(self, batch, training = False):
        # flows shape BxTxCxHxW
        # B: batch
        # C: Channels
        # T: time (frame)
        # H: Frame height
        # W: Frame width

        assert ((not training) or
                 (training and len(self.losses)>0 and self.optim is not None)), "If Training, you should provide losses and optimizer"

        assert (len(batch)==3), "batch should be a tuple of (flows, maks, gt_flows). If you are inferring and don't have gt_flows, then just make batch=(flows, masks, None)"

        flows, masks, gt_flows = batch

        batch_loss = 0
        unitary_losses = [0] * len(self.losses)

        # Remove the batch dimension (for pierrick architecture is needed B to be 1)
        B, T, C, H, W = flows.shape

        # Remove the batch dimension (for pierrick architecture is needed B to be 1)
        flows = flows.view(B * T, C, H, W)
        # masks: 1 inside the hole
        masks = masks.view(B * T, 1, H, W)

        gt_flows = gt_flows.view(B * T, C, H, W)

        # Initial confidence: 1 outside the mask (the hole), 0 inside
        initial_confidence = 1 - 1. * masks
        confidence = initial_confidence.clone()
        gained_confidence = initial_confidence

        new_flow = flows.clone()

        # Get Features from flow
        features = self.encoder(flows)

        step = -1
        while (gained_confidence.sum() > 0) and (step <= self.max_num_steps):
            step += 1
            # print(step)

            if training:
                self.optim.zero_grad()

            current_flow = new_flow.clone().detach()
            current_features = features.clone().detach()

            # Refine features and confidence mask
            new_features, confidence_new = self.__iterative_step(current_features, confidence, current_flow)
            #

            gained_confidence = ((confidence_new > confidence) * confidence_new)

            if gained_confidence.sum() > 0:
                features = current_features * (confidence_new <= confidence) + new_features * (
                            confidence_new > confidence)

                # Get flow from features
                new_flow = self.decoder(features)

                batch_loss = torch.tensor(0)
                i = 0
                for loss, weight in zip(self.losses['losses_list'], self.losses['weights_list']):
                    unitary_loss = torch.tensor(weight) * \
                                   loss(new_flow, mask=gained_confidence, ground_truth=gt_flows)
                    batch_loss = batch_loss + unitary_loss

                    # normalize loss by the number of videos in the test dataset and the bunch of epochs
                    unitary_losses[i] += unitary_loss.item()

                    i += 1

                if training:
                    batch_loss.backward()
                    self.optim.step()

                # mask update before next step
                confidence = confidence_new.clone().detach()

        return batch_loss, unitary_losses


    def load_chk(self, file):

        raise NotImplementedError

    def save_chk(self, file):

        raise NotImplementedError
    '''