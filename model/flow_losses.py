import torch


def mask_L1_loss(flow, gt_flow, mask):
    # L1 loss inside a mask

    pointwise_error = (torch.abs(gt_flow - flow)).mean(dim=1)
    loss = pointwise_error[torch.squeeze(mask) == 1].mean()

    return loss


def L1_loss(flow, gt_flow):
    # L1 loss in all pixels

    pointwise_error = (torch.abs(gt_flow - flow)).mean(dim=1)
    loss = pointwise_error.mean()

    return loss