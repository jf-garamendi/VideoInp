import torch
import torch.nn as nn
import torch.nn.functional


class Min_fbbf_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, flows, device = 'cpu'):
        n, c, h, w = flows.shape

        fw_flow = flows[:, 0:2, :, :]
        bw_flow = flows[:, 2:, :, :]

        loss = 0
        for j in range(1, n - 1):
            loss += torch.mean(torch.min(self.__fb(fw_flow[j, :, :, :], bw_flow[j + 1, :, :, :]),
                                         self.__fb(bw_flow[j, :, :, :], fw_flow[j - 1, :, :, :])))

        return loss

    def __fb(f, b, device):
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