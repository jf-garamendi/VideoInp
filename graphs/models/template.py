import torch
from torch import nn
import torch.nn.functional as F


class ModelTemplate(nn.Module):
    def __init__(self):
        super(ModelTemplate, self).__init__()


    def forward(self, x):
        raise NotImplementedError

    def training_one_epoch(self, batch, losses, optim):
        raise NotImplementedError

    def validation_step(self, batch):

        raise NotImplementedError

    def load_chk(self, file):

        raise NotImplementedError

    def save_chk(self, file):

        raise NotImplementedError
