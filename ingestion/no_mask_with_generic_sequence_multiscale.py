import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

from ingestion.no_mask_with_generic_sequences import No_mask_with_generic_sequences
from os import listdir
from os.path import join
import torch
import random
import configs.folder_structure as folder_structure
from utils.data_io import read_mask, read_frame, read_flow
import numpy as np

class No_mask_with_generic_sequences_multiscale(No_mask_with_generic_sequences):
    def __init__(self, config):
        super().__init__(config)

        self.nLevels=config.n_levels
        #self.video_folders = list(sorted(listdir(join(root_dir, 'level_'+level))))

    def __getitem__(self, idx):
        #returns [coaresest,..,finest]

        # TODO: Split this function into smallest atomic functions (inside these functions do the for loop)
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video_folder = join(self.root_dir, self.video_folders[idx])

        # take a random mask sequence
        n_mask_seq = random.randint(0, len(self.mask_folders)-1)
        masks_folder = join(self.generic_mask_sequences_dir, self.mask_folders[idx])

        #random choose files inside
        # feed with the especified number of frames
        # random the first one and take the next self.number_of_files
        frames_folder = join(video_folder, 'level_1', folder_structure.FRAMES_FOLDER)
        n_files = len(list(listdir(frames_folder)))
        n = min(self.number_of_frames, n_files)
        if n_files> n:
            flow_lower_bound = random.randint(0, n_files - n)
        else:
            flow_lower_bound =0

        flow_upper_bound = flow_lower_bound + n


        n_files = len(list(listdir(masks_folder)))
        n = min(self.number_of_frames, n_files)
        if len(masks_folder) > n:
            mask_lower_bound = random.randint(0, n_files - n)
        else:
            mask_lower_bound = 0

        mask_upper_bound = mask_lower_bound + n

        #----
        pyramid_frames_to_feed = []
        pyramid_flow_to_feed = []
        pyramid_mask_to_feed = []
        pyramid_gt_frames_to_compare = []
        pyramid_gt_flow_to_compare = []
        for level_folder in ['level_'+str(i) for i in reversed(range(self.nLevels))]:

            fwd_flow_folder = join(video_folder, level_folder, folder_structure.FWD_FLOW_FOLDER)
            bwd_flow_folder = join(video_folder, level_folder, folder_structure.BWD_FLOW_FOLDER)
            frames_folder = join(video_folder, level_folder, folder_structure.FRAMES_FOLDER)

            gt_fwd_flow_folder = join(video_folder, level_folder, folder_structure.GT_FWD_FLOW_FOLDER)
            gt_bwd_flow_folder = join(video_folder, level_folder, folder_structure.GT_BWD_FLOW_FOLDER)
            gt_frames_folder = join(video_folder, level_folder, folder_structure.GT_FRAMES_FOLDER)

            mask_files = list(sorted(listdir(masks_folder)))
            frame_files = list(sorted(listdir(frames_folder)))

            fwd_flow_files = list(sorted(listdir(fwd_flow_folder)))
            bwd_flow_files = list(sorted(listdir(bwd_flow_folder)))


            frame_files = frame_files[flow_lower_bound:flow_upper_bound]

            fwd_flow_files = fwd_flow_files[flow_lower_bound:flow_upper_bound]
            bwd_flow_files = bwd_flow_files[flow_lower_bound:flow_upper_bound]

            mask_files = mask_files[mask_lower_bound:mask_upper_bound]

            mask_list = []
            frames_list = []
            flow_list = []
            gt_flow_list = []
            gt_frames_list = []

            for i in range(len(frame_files)):
                #load frames
                frame = read_frame(join(frames_folder, frame_files[i]))

                # load the flow
                fwd_flow = read_flow(join(fwd_flow_folder, fwd_flow_files[i]))
                bwd_flow = read_flow(join(bwd_flow_folder, bwd_flow_files[i]))

                flow = np.concatenate([fwd_flow, bwd_flow], axis=2)

                #Load (or create) mask
                mask_name = join(masks_folder, mask_files[i])
                H = frame.shape[0]
                W = frame.shape[1]
                mask = read_mask(mask_name, background_is='white', H=H, W=W)

                ''' The dilation should be part of the model or at least out of the feeding
                # Dilate and replicate channels in the mask to 4            
                dilated_mask = scipy.ndimage.binary_dilation(mask, iterations=5)
                # Close the small holes inside the foreground objects
                dilated_mask = cv2.morphologyEx(dilated_mask.astype(np.uint8), cv2.MORPH_CLOSE,
                                                 np.ones((21, 21), np.uint8)).astype(np.uint8)
                dilated_mask = scipy.ndimage.binary_fill_holes(dilated_mask).astype(np.uint8)
                '''

                if (i==0) or (i==(len(frame_files)-1)):
                    dilated_mask = mask*0
                else:
                    dilated_mask = mask

                #mask the flow
                masked_flow = flow * np.expand_dims(1 - dilated_mask, -1)

                #mask the frames
                masked_frame = frame * np.expand_dims(1-dilated_mask, -1)

                frames_list.append(masked_frame)
                mask_list.append(dilated_mask)
                flow_list.append(masked_flow)


                gt_frame = read_frame(join(gt_frames_folder, frame_files[i]))
                gt_frames_list.append(gt_frame)

                gt_fwd_flow = read_flow(join(gt_fwd_flow_folder, fwd_flow_files[i]))
                gt_bwd_flow = read_flow(join(gt_bwd_flow_folder, bwd_flow_files[i]))

                gt_flow = np.concatenate([gt_fwd_flow, gt_bwd_flow], axis=2)
                gt_flow_list.append(gt_flow)

            frames_to_feed, flow_to_feed, mask_to_feed, gt_frames_to_compare, gt_flow_to_compare = \
                self.package_data_for_feeding(frames_list, flow_list, mask_list, gt_frames_list, gt_flow_list)

            pyramid_frames_to_feed.append(frames_to_feed)
            pyramid_flow_to_feed.append(flow_to_feed)
            pyramid_mask_to_feed.append(mask_to_feed)
            pyramid_gt_frames_to_compare.append(gt_frames_to_compare)
            pyramid_gt_flow_to_compare.append(gt_flow_to_compare)

        return self.video_folders[idx], (pyramid_frames_to_feed, pyramid_flow_to_feed, pyramid_mask_to_feed, \
               pyramid_gt_frames_to_compare, pyramid_gt_flow_to_compare)



