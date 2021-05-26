import torch
import torch.nn as nn

class L1_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, candidate, ground_truth=None, mask=None):
        pointwise_error = (torch.abs(ground_truth - candidate)).mean(dim=1)

        loss = 0
        if mask is not None:
            loss = pointwise_error[torch.squeeze(mask) > 0].mean()
        else:
            loss = pointwise_error.mean()

        return loss

