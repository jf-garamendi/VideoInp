import torch
import torch.nn.functional

def minfbbf_loss(flows, device):
    def fb(f, b):
        xx, yy = torch.meshgrid(torch.arange(f.shape[-2]), torch.arange(f.shape[-1]))
        ind = torch.stack((yy, xx), dim=-1)
        ind = ind.repeat(f.shape[0], 1, 1, 1).to(device)
        grid = f.permute(( 2, 3, 1)) + ind
        grid = (2 * grid / torch.tensor([f.shape[-1] * 1., f.shape[-2] * 1.]).to(device).view(1, 1, 1, 2)) - 1

        interp = torch.nn.functional.grid_sample(b, grid, mode='bilinear', padding_mode='border', align_corners=False)
        d = torch.norm(interp + f, dim=1)

        return d

    n, c, h, w = flows.shape

    fw_flow = flows[:, 0:2, :, :]
    bw_flow = flows[:, 2:, :, :]

    loss = 0
    for j in range(1, n-1):
        loss += torch.mean(torch.min(fb(fw_flow[j,:,:,:], bw_flow[j+1,:,:,:]), fb(bw_flow[j,:,:,:], fw_flow[j-1,:,:,:])))

    return loss

def TV_loss(candidate, device, eps=1e-3):
    n, c, h, w = candidate.shape

    tv_h = torch.pow(candidate[:,:,1:,:-1]-candidate[:,:,:-1,:-1], 2)
    tv_w = torch.pow(candidate[:,:,:-1,1:] -candidate[:,:,:-1,:-1], 2)

    tv = torch.sqrt(tv_h + tv_w + torch.tensor([eps]).to(device)).sum()

    #Normlize the TV by the number of pixels
    tv = tv / n*c*h*w

    return tv


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