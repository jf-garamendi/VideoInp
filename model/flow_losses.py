import torch

'''
def mask_L1_loss(candidate, ground_truth, mask):
    # L1 loss inside a mask

    pointwise_error = (torch.abs(ground_truth - candidate)).mean(dim=1)
    loss = pointwise_error[torch.squeeze(mask) == 1].mean()

    return loss
'''
def mask_L1_loss(candidate, ground_truth, mask):
    # L1 loss inside a mask

    pointwise_error = torch.sum(torch.abs(ground_truth - candidate),dim=1)
    loss = torch.sum(torch.sum(pointwise_error[torch.squeeze(mask) == 1], dim=0))

    return loss


def L1_loss(flow, gt_flow):
    # L1 loss in all pixels

    pointwise_error = (torch.abs(gt_flow - flow)).mean(dim=1)
    loss = pointwise_error.mean()

    return loss