import torch
import torch.nn as nn

class TV_loss(nn.Module):
    def __init__(self, device = 'cpu'):
        super().__init__()

        self.device = device

    def forward(self, u, **kwargs):
        EPS = torch.tensor(1e-3).to(self.device)

        u_h = u[:, :, 1:, :-1] - u[:, :, :-1, :-1]
        u_w = u[:, :, :-1, 1:] - u[:, :, :-1, :-1]

        u2_h = torch.pow(u_h, 2)
        u2_w = torch.pow(u_w, 2)

        tv = torch.sqrt(u2_h + u2_w + EPS).mean()

        return tv