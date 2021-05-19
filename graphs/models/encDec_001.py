import torch
from torch import nn
import torch.nn.functional as F
from template import ModelTemplate


class encDec_001(ModelTemplate):
    def __init__(self, in_channels = 4, features_channels=32):
        super(encDec_001, self).__init__()

        ##Encoder
        self.conv_enc_1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, 1, 1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU()
        )

        self.conv_enc_2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU()
        )

        ##Decoder
        self.conv_dec_1 = nn.Conv3d(features_channels, 4, kernel_size=1)

    def encode(self, flows):

        out = self.conv_enc_1(flows)
        out = out + self.conv_enc_2(out)

        return out

    def decode(self, features):

        out = self.conv_dec_1(features)

        return out

    def forward(self, x):
        features = self.encode(x)
        out_flows = self.decode(features)

        return out_flows

    def training_one_epoch(self, batch_data, losses, optimizer):
        flows, gt_flows = batch_data


        # Forward Step
        out = self(flows)

        # Loss
        epoch_loss = torch.tensor(0)
        splitted_losses = []
        i = 0
        for loss_fn, weight in zip(losses['losses_list'], losses['weights_list']):
            unitary_loss = torch.tensor(weight) * loss_fn(out, ground_truth=gt_flows)
            epoch_loss = epoch_loss + unitary_loss
            splitted_losses.append(unitary_loss)

            i += 1

        #Backward step
        optimizer.zero_grad()
        epoch_loss.backward()
        optimizer.step()

        return epoch_loss, splitted_losses