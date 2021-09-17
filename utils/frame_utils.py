import torch
from torch.nn import functional

def warp(features, field, device='cuda'):
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

def build_warped_seq(frame_seq, flow, mask_seq):
    # NxCxHxW

    frame_fwd_warped = torch.zeros(frame_seq.shape).to(frame_seq.device) + frame_seq
    frame_bwd_warped = torch.zeros(frame_seq.shape).to(frame_seq.device) + frame_seq

    for i in reversed(range(1, frame_seq.shape[0]-1)):

        frame_tmp_warped = warp(frame_fwd_warped[i+1],flow[i,:2])
        frame_fwd_warped[i] = frame_seq[i]*(1-mask_seq[i]) + mask_seq[i]*frame_tmp_warped


    for i in range(1, frame_seq.shape[0]-1):
        frame_tmp_warped = warp(frame_bwd_warped[i-1], flow[i,2:])
        frame_bwd_warped[i] = frame_seq[i] * (1 - mask_seq[i]) + mask_seq[i] * frame_tmp_warped

    frames_warped = (frame_fwd_warped + frame_bwd_warped) /2

    warped_seq = frame_seq*(1-mask_seq) + frames_warped*mask_seq
    #warped_seq = frame_bwd_warped #frames_warped

    return  warped_seq




def apply_mask(mask, data):
    #mask in 0, 1

    result = mask * data

    return result

