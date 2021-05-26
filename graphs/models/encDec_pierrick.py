import torch
from torch import nn
import torch.nn.functional as F
from .base import BaseTemplate


class EncDec_pierrick(BaseTemplate):
    def __init__(self, in_channels=4, features_channels=32):
        super(EncDec_pierrick, self).__init__()

        ##Encoder
        self.in_c = nn.Conv2d(in_channels, 32, kernel_size=1)

        self.conv_enc_1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU())

        self.conv_enc_2 =nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
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

    def training_one_batch(self, batch):
        # flows shape BxCxTxHxW
        # B: batch
        # C: Channels
        # T: time (frame)
        # H: Frame height
        # W: Frame width

        flows, gt_flows = batch

        # Forward Step
        out = self(flows)

        # Loss
        batch_loss = torch.tensor(0)
        unitary_losses = []
        i = 0
        for loss_fn, weight in zip(self.losses['losses_list'], self.losses['weights_list']):
            unitary_loss = torch.tensor(weight) * loss_fn(out, ground_truth=gt_flows)
            batch_loss = batch_loss + unitary_loss
            unitary_losses.append(unitary_loss)

            i += 1

        # Backward step
        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        return batch_loss, unitary_losses

    def validating_one_batch(self, batch):
        flows, gt_flows = batch

        # Forward Step
        out = self(flows)

        # Loss
        batch_loss = torch.tensor(0)
        unitary_losses = []
        i = 0
        for loss_fn, weight in zip(self.losses['losses_list'], self.losses['weights_list']):
            unitary_loss = torch.tensor(weight) * loss_fn(out, ground_truth=gt_flows)
            batch_loss = batch_loss + unitary_loss
            unitary_losses.append(unitary_loss)

            i += 1

        return batch_loss, unitary_losses

    def inferring_one_batch(self, batch):

        return self(batch)
