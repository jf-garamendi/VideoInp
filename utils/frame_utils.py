import torch
from torch.nn import functional

def warp(features, field, device):
    # features: size (CxHxW)
    # field: size (2xHxW)

    C, H, W = features.shape

    # Grid for warping
    xx, yy = torch.meshgrid(torch.arange(H), torch.arange(W))
    ind = torch.stack((yy, xx), dim=-1).to(device)

    field = field.permute((1, 2, 0)) + ind
    field = torch.unsqueeze(field, 0)

    # Normalize the coordinates to the square [-1,1]
    field = (2 * field / torch.tensor([W, H]).view(1, 1, 1, 2).to(device)) - 1

    # warp ## FORWARD ##
    features2warp = torch.unsqueeze(features, 0)
    warped_features = functional.grid_sample(features2warp, field,
                                          mode='bilinear', padding_mode='border',
                                          align_corners=False)
    warped_features = torch.squeeze(warped_features)

    return warped_features

def apply_mask(mask, data):
    #mask in 0, 1

    result = mask * data

    return result

