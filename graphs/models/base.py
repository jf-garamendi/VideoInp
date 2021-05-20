import torch
from torch import nn
import torch.nn.functional as F


class BaseTemplate(nn.Module):
    def __init__(self):
        super(BaseTemplate, self).__init__()


    def forward(self, x):
        raise NotImplementedError

    def training_one_batch(self, batch):

        # call self.inference_step()

        # Compute losses

        # return losses

        raise NotImplementedError

    def validating_one_batch(self, batch):
        # call self.inference_step()

        # Compute losses

        # return losses

        raise NotImplementedError

    def inferring_one_batch(self, batch):

        # most of the times this is self(batch) (i.e. calling to forward())

        raise NotImplementedError

    def load_chk(self, file):

        raise NotImplementedError

    def save_chk(self, file):

        raise NotImplementedError
