import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import os
from utils.flow_viz import read_flow_from_file
import numpy as np

#constants
# name of the folders inside the sequences root folder
FRAMES_FOLDER = "frames"
MASKS_FOLDER = "masks"
GT_FWD_FLOW_FOLDER = "gt_fwd_flow"
GT_BWD_FLOW_FOLDER = "gt_bwd_flow"
GT_FRAMES_FOLDER = "gtFrames"


class DataIngestion(Dataset):
    def __init__(self, root_dir, n_frames=1, flow_on_the_fly=False, training = True):
        # root_dir:
        # nFrames:
        # flow_on_the_fly:
        # training: If training is True, then we read the Ground Truth

        self.root_dir = root_dir
        self.n_frames = n_frames
        self.training = training

        self.video_folders = list(sorted(os.listdir(root_dir)))


    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video_folder = os.path.join(self.root_dir, self.video_folders[idx])

        mask_files = list(sorted(os.listdir(os.path.join(video_folder, MASKS_FOLDER))))

        fwd_flow_files = list(sorted(os.listdir(os.path.join(video_folder, GT_FWD_FLOW_FOLDER))))
        bwd_flow_files = list(sorted(os.listdir(os.path.join(video_folder, GT_BWD_FLOW_FOLDER))))

        mask_list = []
        flow_list = []
        gt_flow_list = []

        for i in range(len(mask_files)):
            # build the complete file names
            mask_name = mask_files[i]

            # load the data
            mask = io.imread(mask_name)

            fwd_flow = read_flow_from_file(fwd_flow_files[i])
            bwd_flow = read_flow_from_file(bwd_flow_files[i])

            #Change de levels to [-1,1] range in order to feed the net (keep the mask at levels [0,1]
            fwd_flow = self.level_to_range(fwd_flow, -1, 1)
            bwd_flow = self.level_to_range(bwd_flow, -1, 1)

            #Mask the fwd_flow and bwd_flow
            masked_fwd_flow = self.apply_mask(mask, fwd_flow)
            masked_bwd_flow = self.apply_mask(mask, bwd_flow)

            flow = np.stack([masked_fwd_flow, masked_bwd_flow], axis=2)  # TODO: Revisar las dimensiones

            mask_list.append(mask)
            flow_list.append(flow)

            if self.training:
                gt_flow = np.stack([fwd_flow, bwd_flow], axis=2)  # TODO: Revisar las dimensiones
                gt_flow_list.append(gt_flow)


        # To numpy arrays
        video_masks = np.array(mask_list)
        video_flow = np.array(flow_list)
        video_gt_flow = np.array((gt_flow_list))

        # To Tensors
        video_masks = torch.from_numpy(video_masks).contiguous().float()
        video_flow = torch.from_numpy(video_flow).contiguous().float()
        video_gt_flow = torch.from_numpy(video_gt_flow).contiguous().float()

        sample = {
            'mask': video_masks,
            'flow' : video_flow,
            'gt_frame' : video_gt_flow
            }

        return sample

    def level_to_range(self, frame, lower_bound=0, upper_bound=1):
        #Image in format HxWxRGB
        # normalize to range [low,1]

        # Shift the levels to the left such a way the min will be lower_bound
        mn = frame.min()
        frame = frame - mn - lower_bound

        # Scale the levels to the right such a way the max will be upper_bound
        mx = frame.max()
        frame = (frame / mx) * (upper_bound-lower_bound)

        return frame








