import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

from ingestion.VideoInp_DataSet import VideoInp_DataSet
from os import listdir
from os.path import join, isfile
import torch
import random
import configs.folder_structure as folder_structure
from utils.data_io import read_mask, read_frame, read_flow
import numpy as np

class VideoInp_Dataset_Multiscale(VideoInp_DataSet):
    def __init__(self, config):
        # masks can be:
        #  * same file mask (defined in mask_file) for all pixels
        #  * each video has its own set of masks inside folder join(video_folder, folder_structure.MASKS_FOLDER)

        super().__init__(config.root_dir, GT=True, number_of_frames=config.number_of_frames)

        self.mask_file_for_all_frames = None
        if  hasattr(config, "mask"):
            if isfile(config.mask):
                self.mask_file_for_all_frames = config.mask

        self.nLevels = config.n_levels


    def __getitem__(self, idx):
        #returns [coaresest,..,finest]

        # TODO: Split this function into smallest atomic functions (inside these functions do the for loop)
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video_folder = join(self.root_dir, self.video_folders[idx])
        masks_folder = join(video_folder, folder_structure.MASKS_FOLDER)

        #random choose files inside
        # feed with the especified number of frames
        # random the first one and take the next self.number_of_files
        frames_folder = join(video_folder, 'level_1', folder_structure.FRAMES_FOLDER)
        n_files = len(list(listdir(frames_folder)))
        n = min(self.number_of_frames, n_files)
        if n_files> n:
            #flow_lower_bound = random.randint(0, n_files - n)
            #debug
            lower_bound = 0
        else:
            lower_bound = 0

        upper_bound = lower_bound + n


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

            if self.mask_file_for_all_frames is None:
                mask_files = list(sorted(listdir(masks_folder)))
                mask_files = mask_files[lower_bound:upper_bound]
            else:
                mask_files = [self.mask_file_for_all_frames] * len(range(lower_bound, upper_bound))

            frame_files = list(sorted(listdir(frames_folder)))
            frame_files = frame_files[lower_bound:upper_bound]

            fwd_flow_files = list(sorted(listdir(fwd_flow_folder)))
            bwd_flow_files = list(sorted(listdir(bwd_flow_folder)))
            fwd_flow_files = fwd_flow_files[lower_bound:upper_bound]
            bwd_flow_files = bwd_flow_files[lower_bound:upper_bound]



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

                #Load mask
                mask_name = join(masks_folder, mask_files[i])
                H = frame.shape[0]
                W = frame.shape[1]
                mask = read_mask(mask_name, H=H, W=W)

                ''' The dilation should be part of the model or at least out of the feeding
                # Dilate and replicate channels in the mask to 4            
                dilated_mask = scipy.ndimage.binary_dilation(mask, iterations=5)
                # Close the small holes inside the foreground objects
                dilated_mask = cv2.morphologyEx(dilated_mask.astype(np.uint8), cv2.MORPH_CLOSE,
                                                 np.ones((21, 21), np.uint8)).astype(np.uint8)
                dilated_mask = scipy.ndimage.binary_fill_holes(dilated_mask).astype(np.uint8)
                '''

                #First and last frame without masks
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



