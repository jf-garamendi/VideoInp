import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

import argparse
from glob import glob
from os.path import join
from PIL import Image
import numpy as np
from RAFT.utils.utils import initialize_RAFT, calculate_flow
from utils.data_io import create_dir, writeFlow


import constants
import torch

def main(args):
    # List the videos in the root folder
    video_folders = list(sorted(os.listdir(args.in_root_dir)))

    #Create root desteny folder
    create_dir(args.out_dir)

   # Prcess frames
    for video_path in video_folders:
        out_dir = join(args.out_dir, video_path)

        video_path = join(args.in_root_dir, video_path)
        frame_filename_list = glob(join(video_path, constants.FRAMES_FOLDER, '*.png')) +\
                              glob(join(video_path, constants.MASKS_FOLDER, '*.jpg'))

        frame_filename_list = sorted(frame_filename_list)

        gt_frames = []
        for filename in frame_filename_list:
            f = np.array(Image.open(filename)).astype(np.uint8)

            gt_frames.append(f)

        #create the mask
        masked_frames = []
        #if args.masking_mode == "template":
        masks, masked_frames = create_template_mask_data(gt_frames, args.template_mask)

        # Compute optical flow, if needed
        if args.compute_RAFT_flow:
            gt_fwd_flow, gt_bwd_flow = create_RAFT_flow(gt_frames, args)

            if args.apply_mask_before:
                masked_fwd_flow, masked_bwd_flow = create_RAFT_flow(masked_frames, args)
            else:
                masked_fwd_flow = []
                masked_bwd_flow = []

                for frame_fwd, frame_bwd, mask in zip(gt_fwd_flow, gt_bwd_flow, masks):
                    masked_fwd_flow.append(frame_fwd * np.expand_dims(1-mask, -1))
                    masked_bwd_flow.append(frame_bwd * np.expand_dims(1-mask, -1))



        save_data(masks, masked_frames, masked_fwd_flow, masked_bwd_flow, gt_frames, gt_fwd_flow, gt_bwd_flow, out_dir)


def save_data(masks, masked_frames, fwd_flow, bwd_flow, gt_frames, gt_fwd_flow, gt_bwd_flow, out_dir):

    folders = {
        "mask_dir": join(out_dir, constants.MASKS_FOLDER),
        "frame_dir" : join(out_dir, constants.FRAMES_FOLDER),
        "fwd_flow_dir" : join(out_dir, constants.FWD_FLOW_FOLDER),
        "bwd_flow_dir" : join(out_dir, constants.BWD_FLOW_FOLDER),
        "gt_frame_dir" : join(out_dir, constants.GT_FRAMES_FOLDER),
        "gt_fwd_flow_dir" : join(out_dir, constants.GT_FWD_FLOW_FOLDER),
        "gt_bwd_flow_dir" : join(out_dir, constants.GT_BWD_FLOW_FOLDER)
    }

    for _, value in folders.items():
        create_dir(value)

    # saving masks, frames and flow
    for i,(m, fr, fwd, bwd) in enumerate(zip(masks, masked_frames, fwd_flow, bwd_flow)):
        m = Image.fromarray(m*255)
        name = join(folders["mask_dir"], '%04d.png' % i)
        m.save(name)
        print("saved mask in " + name)

        fr = Image.fromarray(fr)
        name = join(folders["frame_dir"], '%04d.jpg' % i)
        fr.save(name)
        print("saved masked frame in " + name)

        name = join(folders["fwd_flow_dir"], '%04d.flo' % i)
        writeFlow(name, fwd)
        print("saved masked forward flow in " + name)

        name = join(folders["bwd_flow_dir"], '%04d.flo' % i)
        writeFlow(name, bwd)
        print("saved masked backward flow in in " + name)

    # Saving Ground Truth
    for i, (fr, fwd, bwd) in enumerate(zip(gt_frames, gt_fwd_flow, gt_bwd_flow)):

        fr = Image.fromarray(fr)
        name = join(folders["gt_frame_dir"], '%04d.jpg' % i)
        fr.save(name)
        print("saved ground truth of frame in " + name)

        name = join(folders["gt_fwd_flow_dir"], '%04d.flo' % i)
        writeFlow(name, fwd)
        print("saved ground truth of forward flow in " + name)

        name = join(folders["gt_bwd_flow_dir"], '%04d.flo' % i)
        writeFlow(name, bwd)
        print("saved ground truth of backward flow in " + name)

    # Debug purposes BORRAR
    # from utils.io import save_flow_and_img
    # save_flow_and_img(fwd_flow, folder_flow='./borrar_fwd', folder_img='./borrar_fwd_png')
    # save_flow_and_img(bwd_flow, folder_flow='./borrar_bwd', folder_img='./borrar_bwd_png')
    # save_flow_and_img(gt_fwd_flow, folder_flow='./borrar_fwd_img', folder_img='./borrar_fwd_png')
    # save_flow_and_img(gt_bwd_flow, folder_flow='./borrar_bwd_img', folder_img='./borrar_bwd_png')

def create_template_mask_data(frame_list, path_to_mask):
    H, W, _ = frame_list[0].shape

    pil_mask = Image.open(path_to_mask).convert('L')

    mask = pil_mask.resize((W,H))
    mask = np.array(mask).astype(np.uint8)
    mask = (mask > 0).astype(np.uint8)

    masked_frames = []
    masks = []
    for frame in frame_list:
        masked_f = frame * np.expand_dims(1-mask, -1)

        masked_frames.append(masked_f)
        masks.append(mask)

    return masks, masked_frames


def create_RAFT_flow(frame_list, RAFT_args):
    # frame_list: each element of size RGB x H x W

    # To dimensions: RGB x H x W (channel first style)
    # and convert to tensor
    torch_frame_list =[]
    for i, frame in enumerate(frame_list):
        torch_frame_list.append(torch.from_numpy(frame.astype(np.uint8)).permute(2, 0, 1).float())

    video = torch.stack(torch_frame_list, dim=0)
    video = video.to('cuda')

    nFrames, _, imgH, imgW = video.shape

    #Flow model
    RAFT_model = initialize_RAFT(RAFT_args)

    # Calcutes the flow. Notice that this flow is computed  on the non-masked video
    fwd_flow = calculate_flow(RAFT_model, video, 'forward')
    # add the lid to the final
    fwd_flow = fwd_flow + [np.zeros((imgH, imgW, 2))]

    bwd_flow = calculate_flow(RAFT_model, video, 'backward')
    # add the lid to the beginning
    bwd_flow = [np.zeros((imgH, imgW, 2))] + bwd_flow
    print('\nFinish flow prediction.')
    # END calculate the flow

    return  fwd_flow, bwd_flow

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_root_dir', default=None,
                        help='Path to the root folder where the videos are saved')
    parser.add_argument('--out_dir', default=None,
                        help='Path to the root folder where the dataset will be saved')
    parser.add_argument('--masking_mode', default=None,
                        help='mode of making the masks. Modes are Blah, blah, blah....')
    parser.add_argument('--template_mask', default=None,
                        help='Path to the image used as template for masking')
    parser.add_argument('--compute_RAFT_flow', action='store_true',
                        help='Whether compute the optical flow (using RAFT) or not')
    parser.add_argument('--apply_mask_before', action='store_true',
                        help='If active, apply mask to the frames before computing the optical flow.')
    parser.add_argument('--apply_mask_after', action='store_true',
                        help='If active, apply mask to the flow after computing the optical flow.')

    # RAFT
    parser.add_argument('--opticalFlow_model', default='../weight/raft-things.pth',
                        help="Path to the RAFT checkpoint for computing the Optical flow.")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    args = parser.parse_args()

    main(args)