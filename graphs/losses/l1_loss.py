import torch
import torch.nn as nn

class L1_loss(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()

        self.device = device

    def forward(self, candidate, ground_truth=None, mask=None, **kwargs):
        # dim=1 is the channels dimension, as it is always 4, the mean along that dimension doesn't affect de |·|_1 behaviour
        pointwise_error = (torch.abs(ground_truth - candidate)).mean(dim=1)

        loss = 0
        if (mask is not None) :
            loss = pointwise_error[torch.squeeze(mask) > 0].mean()
        else:
            loss = pointwise_error.mean()

        return loss

