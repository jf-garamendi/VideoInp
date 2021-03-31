import torch
from torch.utils.data import Dataset, DataLoader
from os import listdir
from os.path import join
from utils.data_io import read_flow
import numpy as np
from utils.frame_utils import  apply_mask
from utils.data_io import load_mask
import constants
import random
import scipy.ndimage
import cv2
#For reproducibility
np.random.seed(2021)
random.seed(2021)
torch.manual_seed(2021)
from PIL import Image

class VideoInp_DataSet(Dataset):
    def __init__(self, root_dir, training = True, random_mask_on_the_fly= False, flow_on_the_fly=False):
        # root_dir:
        # nFrames: NOT IMPLEMENTED
        # random_mask_on_the_fly: If True, then does not read the mask from file and generate a random square mask
        # flow_on_the_fly: NOT IMPLEMENTED. Does not read the optical flow from file and Computes it flow on the fly
        # training: If training is True, then we read the Ground Truth

        self.root_dir = root_dir
        self.training = training

        self.random_mask_on_the_fly = random_mask_on_the_fly

        self.video_folders = list(sorted(listdir(root_dir)))


    def __len__(self):
        return len(self.video_folders)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video_folder = join(self.root_dir, self.video_folders[idx])

        masks_folder = join(video_folder, constants.MASKS_FOLDER)
        fwd_flow_folder = join(video_folder, constants.FWD_FLOW_FOLDER)
        bwd_flow_folder = join(video_folder, constants.BWD_FLOW_FOLDER)

        gt_fwd_flow_folder = join(video_folder, constants.GT_FWD_FLOW_FOLDER)
        gt_bwd_flow_folder = join(video_folder, constants.GT_BWD_FLOW_FOLDER)

        mask_files = list(sorted(listdir(masks_folder)))

        fwd_flow_files = list(sorted(listdir(fwd_flow_folder)))
        bwd_flow_files = list(sorted(listdir(bwd_flow_folder)))

        mask_list = []
        flow_list = []
        gt_flow_list = []

        for i in range(len(fwd_flow_files)):
            # build the complete file names
            mask_name = join(masks_folder, mask_files[i])

            # load the flow
            fwd_flow = read_flow(join(fwd_flow_folder, fwd_flow_files[i]))
            bwd_flow = read_flow(join(bwd_flow_folder, bwd_flow_files[i]))

            flow = np.concatenate([fwd_flow, bwd_flow], axis=2)

            #Load (or create) mask
            mask = 0
            if self.random_mask_on_the_fly:
                mask = self.compute_random_mask(fwd_flow.shape[0], fwd_flow.shape[1], n_squares = 1)
                ''' Debug
                m_pil = Image.fromarray(255 * mask)
                if m_pil.mode != 'RGB':
                    m_pil = m_pil.convert('RGB')
                m_pil.save('borrar_mask.png')
                '''
            else:
                mask = load_mask(mask_name)

            # Dilate and replicate channels in the mask to 4
            #dilated_mask = scipy.ndimage.binary_dilation(mask, iterations=15)
            dilated_mask = scipy.ndimage.binary_dilation(mask, iterations=5)
            # Close the small holes inside the foreground objects
            dilated_mask = cv2.morphologyEx(dilated_mask.astype(np.uint8), cv2.MORPH_CLOSE,
                                             np.ones((21, 21), np.uint8)).astype(np.uint8)
            dilated_mask = scipy.ndimage.binary_fill_holes(dilated_mask).astype(np.uint8)

            #mask the flow
            masked_flow = flow * np.expand_dims(1 - dilated_mask, -1)

            mask_list.append(dilated_mask)
            flow_list.append(masked_flow)

            if self.training:
                gt_fwd_flow = read_flow(join(gt_fwd_flow_folder, fwd_flow_files[i]))
                gt_bwd_flow = read_flow(join(gt_bwd_flow_folder, bwd_flow_files[i]))

                gt_flow = np.concatenate([gt_fwd_flow, gt_bwd_flow], axis=2)
                gt_flow_list.append(gt_flow)

        flow_to_feed, mask_to_feed, gt_flow_to_compare = self.package_data_for_feeding(flow_list, mask_list, gt_flow_list)

        return flow_to_feed, mask_to_feed, gt_flow_to_compare

    #def compute_random_mask(self,  max_mask_W, max_mask_H, W_frame, H_frame ):
    def compute_random_mask(self, H_frame, W_frame, n_squares=1):
        # TODO: Make max_H, and max_W parameters
        #The test mask has a rtio of H/9, W/13
        max_H = H_frame/8
        max_W = W_frame/10

        mask = np.zeros((H_frame, W_frame)).astype(np.uint8)

        for i in range(n_squares):
            top_left = (np.random.randint(0,H_frame), np.random.randint(0,W_frame))
            bottom_right = (
                np.random.randint(top_left[0], min(H_frame, top_left[0]+max_H)),
                np.random.randint(top_left[1], min(W_frame, top_left[1]+max_W))
            )

            mask[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = 1


        return mask

    def package_data_for_feeding(self, flow_list, mask_list, gt_flow_list):
        # mask_list
        # flow_list: list, each element HxWxC, where C=4 --> two first forward, two last bwd
        # gt_flow_list: list, each element HxWxC, where C=4 --> two first forward, two last bwd

        flow = np.stack(flow_list)
        mask = np.stack(mask_list)
        gt_flow = np.stack(gt_flow_list)

        flow = torch.from_numpy(flow).permute(0, 3, 1, 2).contiguous().float()
        mask = torch.from_numpy(mask).view(mask.shape[0], 1, mask.shape[1], mask.shape[2]).contiguous().float()
        gt_flow = torch.from_numpy(gt_flow).permute(0, 3, 1, 2).contiguous().float()

        return flow, mask, gt_flow

    ''' THIS SHOULD BE NEEDED IN THE FUTURE, intercala las mascaras y los optical flow
    def package_data_for_feeding(self, mask_list, flow_list, gt_flow_list):
        # mask_list
        # flow_list: list, each element HxWxC, where C=4 --> two first forward, two last bwd
        # gt_flow_list: list, each element HxWxC, where C=4 --> two first forward, two last bwd

        # Alternate mask and flow
        input_data = []
        for mask, flow in zip(mask_list, flow_list):

            # concatenate in the channel dimension (the last one)
            mask = np.expand_dims(mask, axis=2)
            data_element = np.concatenate((mask,flow), axis=2)

            input_data.append(data_element)

        input_data = np.stack(input_data)
        gt_flow = np.stack(gt_flow_list)

        input_data = torch.from_numpy(input_data).permute(3, 0, 1, 2).contiguous().float()
        gt_flow = torch.from_numpy(gt_flow).permute(3, 0, 1, 2).contiguous().float()


        return input_data,  gt_flow
    '''

    def level_to_range(self, frame, lower_bound=0, upper_bound=1):
        #TODO: Debe ir al modelo

        #Image in format HxWxRGB
        # normalize to range [low,1]

        # Shift the levels to the left such a way the min will be lower_bound
        mn = frame.min()
        frame = frame - mn - lower_bound

        # Scale the levels to the right such a way the max will be upper_bound
        mx = frame.max()
        frame = (frame / mx) * (upper_bound-lower_bound)

        return frame