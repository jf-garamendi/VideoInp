import torch
import torch.nn.functional


def minfbbf(flows, device= 'cpu', **kwargs):

    def fb(f, b):
        C, H, W = f.shape
        xx, yy = torch.meshgrid(torch.arange(H), torch.arange(W))
        ind = torch.stack((yy, xx), dim=-1).to(device)


        grid = f.permute((1, 2, 0)) + ind
        grid = torch.unsqueeze(grid, 0)

        # Normalize coordinates to the square [-1, 1]
        grid = (2*grid / torch.tensor([W, H]).view(1,1,1,2).to(device))-1

        b2warp = torch.unsqueeze(b, 0)
        interp = torch.nn.functional.grid_sample(b2warp, grid,
                                                 mode='bilinear', padding_mode='border',
                                                 align_corners=False)
        warped_b = torch.squeeze(interp)
        d = torch.norm(warped_b + f, dim=0)

        return d

    n, c, h, w = flows.shape

    fw_flow = flows[:, 0:2, :, :]
    bw_flow = flows[:, 2:, :, :]

    loss = 0
    for j in range(1, n - 1):
        loss += torch.mean(torch.min(fb(fw_flow[j, :, :, :], bw_flow[j + 1, :, :, :]),
                                     fb(bw_flow[j, :, :, :], fw_flow[j - 1, :, :, :])))

    return loss


def TV(u, **kwargs):
    EPS =1e-3

    u_h = u[:, :, 1:, :-1] - u[:, :, :-1, :-1]
    u_w = u[:, :, :-1, 1:] - u[:, :, :-1, :-1]

    u2_h = torch.pow(u_h, 2)
    u2_w = torch.pow(u_w, 2)


    # torch manage the division by zero (uses a regularized version of sqrt)
    tv = torch.sqrt(u2_h + u2_w+ EPS).mean()

    return tv


def L1(candidate, ground_truth=None, mask=None, **kwargs):
    loss = 0

    pointwise_error = (torch.abs(ground_truth - candidate)).mean(dim=1)

    if mask is not None:
        loss = pointwise_error[torch.squeeze(mask) == 1].mean()
    else:
        loss = pointwise_error.mean()

    return loss
