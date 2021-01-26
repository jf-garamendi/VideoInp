import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

import argparse
from os.path import join
import cv2
import glob
#import copy
import numpy as np
import torch
#import imageio
from PIL import Image
import scipy.ndimage
#from skimage.feature import canny
#import torchvision.transforms.functional as F

from RAFT import utils
from RAFT import RAFT

#import utils.region_fill as rf
#from utils.Poisson_blend import Poisson_blend
#from utils.Poisson_blend_img import Poisson_blend_img
#from get_flowNN import get_flowNN
#from get_flowNN_gradient import get_flowNN_gradient
#from utils.common_utils import flow_edge
#from spatial_inpaint import spatial_inpaint
#from frame_inpaint import DeepFillv1
#from edgeconnect.networks import EdgeGenerator_

from models.iterative import Flow2features, Features2flow, Res_Update3, Res_Update2
from utils import flow_viz, frame_utils
import utils.region_fill as rf

from tqdm import tqdm

def create_dir(dir):
    """Creates a directory if not exist.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)

def initialize_RAFT(args):
    """Initializes the RAFT model.
    """
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.opticalFlow_model))

    model = model.module
    model.to('cuda')
    model.eval()

    return model

def initialize_encoder(weight_path):
    model = Flow2features().cpu().eval()

    # load weights
    flow2F_ckpt_dict = torch.load(weight_path)
    model.load_state_dict(flow2F_ckpt_dict['flow2F'], strict=True)

    return model

def initialize_decoder(weight_path):
    # decoder Features to Flow
    model = Features2flow().cpu().eval()

    # load weights
    F2flow_ckpt_dict = torch.load(weight_path)
    model.load_state_dict(F2flow_ckpt_dict['F2flow'], strict=True)

    return model
def initialize_update(number, kind_update, weight_path):
    model = None
    if number == 2:
        model  = Res_Update2(update=kind_update)
    elif number == 3:
        model = Res_Update3(update=kind_update)
    elif number == 4:
        model = Res_Update4(update=kind_update)

    model = model.cpu().eval()

    update_ckpt_dict = torch.load(weight_path)
    model.load_state_dict(update_ckpt_dict['update'], strict=True)

    return model


def gradient_mask(mask):

    gradient_mask = np.logical_or.reduce((mask,
        np.concatenate((mask[1:, :], np.zeros((1, mask.shape[1]), dtype=np.bool)), axis=0),
        np.concatenate((mask[:, 1:], np.zeros((mask.shape[0], 1), dtype=np.bool)), axis=1)))

    return gradient_mask

def calculate_flow(model, video, mode):
    """Calculates optical flow.
    """
    if mode not in ['forward', 'backward']:
        raise NotImplementedError

    nFrame, _, imgH, imgW = video.shape
    #Flow = np.empty(((imgH, imgW, 2, 0)), dtype=np.float32)
    #Flow = np.empty(((nFrame, 2, imgH, imgW)), dtype=np.float32)
    Flow=[]


    with torch.no_grad():
        for i in range(video.shape[0] - 1):
            print("Calculating {0} flow {1:2d} <---> {2:2d}".format(mode, i, i + 1), '\r', end='')
            if mode == 'forward':
                # Flow i -> i + 1
                image1 = video[i, None]
                image2 = video[i + 1, None]
            elif mode == 'backward':
                # Flow i + 1 -> i
                image1 = video[i + 1, None]
                image2 = video[i, None]
            else:
                raise NotImplementedError

            _, flow = model(image1, image2, iters=20, test_mode=True)
            flow = flow[0].permute(1, 2, 0).cpu().numpy()
            #Flow = np.concatenate((Flow, flow[..., None]), axis=-1)
            Flow.append(flow)

            ''' NO NECESARIO, BORRAR
            # Flow visualization.
            if outroot is not None:
                flow_img = flow_viz.flow_to_image(flow)
                flow_img = Image.fromarray(flow_img)

                # Saves the flow and flow_img.
                flow_img.save(os.path.join(outroot, 'GT_flow', mode + '_png', '%05d.png'%i))
                frame_utils.writeFlow(os.path.join(outroot, 'GT_flow', mode + '_flo', '%05d.flo' % i), flow)
            '''

    return Flow

def load_video_frames(video_path):
    # Loads frames.
    frame_filename_list = glob.glob(os.path.join(video_path, '*.png')) + \
                          glob.glob(os.path.join(video_path, '*.jpg'))

    video = []
    for filename in sorted(frame_filename_list):
        video.append(torch.from_numpy(np.array(Image.open(filename)).astype(np.uint8)).permute(2, 0, 1).float())

    video = torch.stack(video, dim=0)

    return video


def save_flow(flow, folder):
    #flow: list of matrices (each matrix is the flow of a frame) of size (H,W,C)

    folder_flow = os.path.join(folder, 'flow_flo')
    folder_png = os.path.join(folder, 'flow_png')

    create_dir(folder_flow)
    create_dir(folder_png)

    for i in tqdm(range(len(flow))):
        flow_frame = flow[i]

        flow_img = flow_viz.flow_to_image(flow_frame)
        flow_img = Image.fromarray(flow_img)

        # Saves the flow and flow_img.
        frame_utils.writeFlow(os.path.join(folder_flow, '%05d.flo' % i), flow_frame)
        flow_img.save(os.path.join(folder_png, '%05d.png' % i))

def object_removal_seamless(args):

    # Flow model.
    RAFT_model = initialize_RAFT(args)

    # Loads frames.
    video = load_video_frames(args.video_path)
    video = video.to('cuda')

    # Calcutes the flow. Notice that this flow is computed  on the non-masked video
    forward_flow = calculate_flow(RAFT_model, video, 'forward')
    # add the lid to the final
    #forward_flow = np.concatenate((forward_flow, np.zeros((imgH, imgW, 2,1))), axis=3)

    backward_flow = calculate_flow(RAFT_model, video, 'backward')
    # add the lid to the beginning
    #backward_flow = np.concatenate((np.zeros((imgH, imgW, 2,1)), backward_flow), axis=3)
    print('\nFinish flow prediction.')
    # END calculate the flow

    if args.verbose:
        print ('Saving computed flow (without masking) into  ' + args.verbose_path)
        save_flow(forward_flow, join(args.verbose_path, 'GT_forward_flow'))
        save_flow(backward_flow, join(args.verbose_path, 'GT_backward_flow'))

    # free GPU memory,
    video = video.to('cpu')

    #Load the masks  and mask the flow<--TODO: Create a function
    mask_filename_list = glob.glob(os.path.join(args.path_mask, '*.png')) + \
                    glob.glob(os.path.join(args.path_mask, '*.jpg'))

    mask_set = [] #mask_Set in pierrick's code
    forward_masked_flow = []
    backward_masked_flow = []
    for (i,filename) in enumerate(sorted(mask_filename_list)):
        mask_img = np.array(Image.open(filename).convert('L'))


        # Dilate 15 pixel so that all known pixel is trustworthy
        flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=15)
        # Close the small holes inside the foreground objects
        flow_mask_img = cv2.morphologyEx(flow_mask_img.astype(np.uint8), cv2.MORPH_CLOSE,
                                         np.ones((21, 21), np.uint8)).astype(np.uint8)
        flow_mask_img = scipy.ndimage.binary_fill_holes(flow_mask_img).astype(np.uint8)

        # Adapt to pierrick's format
        flow_mask_img = np.expand_dims(flow_mask_img, axis=2)

        # mask the forward flow
        if i < len(forward_flow):
            flow_frame = forward_flow[i]
            forward_masked_flow.append(flow_frame * (1. - flow_mask_img))


        # mask the backward flow
        if i > 0:
            flow_frame = backward_flow[i-1]
            backward_masked_flow.append(flow_frame * (1. - flow_mask_img))

        mask_set.append(np.concatenate((flow_mask_img, flow_mask_img), axis=2))

    if args.verbose:
        folder = join(args.verbose_path, 'masked_forward_flow')
        print ('Saving initial masked forward flow into ' + folder)
        save_flow(forward_masked_flow, folder)

        folder = join(args.verbose_path, 'masked_backward_flow')
        print('Saving initial masked backward flow into ' + folder)
        save_flow(backward_masked_flow, folder)

    # END Load the masks

    # Completes the flow.
    completed_forward_flow, completed_backward_flow = complete_flow(args, forward_masked_flow, backward_masked_flow, mask_set)
    print('\nFinish flow completion.')

def smooth_flow_fill(flow, masks):
    for i in tqdm(range(len(flow))):
        flow[i][:,:,0] =  rf.regionfill(flow[i][:, :, 0], masks[i][:,:,0])
        flow[i][:, :, 1] = rf.regionfill(flow[i][:, :, 1], masks[i][:, :, 1])

    return flow

def complete_flow(args, forward_flow, backward_flow, masks):

    #Initialize the flow inside the mask with a smooth interpolation
    print("Initializing flow inside the inpainting hole")
    forward_flow  = smooth_flow_fill(forward_flow, masks)
    backward_flow = smooth_flow_fill(backward_flow, masks)


    if args.verbose:
        folder = join(args.verbose_path, 'initial_forward_flow')
        print ('Saving initialized forward flow into ' + folder)
        save_flow(forward_flow, folder)

        folder = join(args.verbose_path, 'initial_backward_flow')
        print('Saving initial initial backward flow into ' + folder)
        save_flow(backward_flow, folder)

    masks = np.expand_dims(np.stack(masks, axis=3), axis=0)
    forward_flow = np.expand_dims(np.stack(forward_flow, axis=3), axis=0)
    backward_flow = np.expand_dims(np.stack(backward_flow, axis=3), axis=0)

    # re-order into the Pierrick's ordering
    masks = torch.from_numpy(masks).permute(0,4,3,1,2).contiguous().float()
    forward_flow = torch.from_numpy(forward_flow).permute(0,4,3,1,2).contiguous().float()
    backward_flow = torch.from_numpy(backward_flow).permute(0,4,3,1,2).contiguous().float()

    forward_flow.cpu()
    backward_flow.cpu()
    masks.cpu()

    B, N, C, H, W = forward_flow.shape

    # mask indicating the missing region in the video.
    hole = 1 - masks
    initial_confidence = 1 - 1. * masks
    current_confidence = 1. * initial_confidence

    xx, yy = torch.meshgrid(torch.arange(H), torch.arange(W))
    ind = torch.stack((yy, xx), dim=-1)
    ind = ind.repeat(B, 1, 1, 1)

    with torch.no_grad():
        ###
        ### Build the complete model and Load the weights
        # encoder Flow to Features
        flow2F = initialize_encoder(args.flow2features_weights)

        # decoder Features to Flow
        F2flow = initialize_decoder(args.features2flow_weights)

        # Update net
        update_net = initialize_update(int(args.update_number_model), args.kind_update, args.update_weights)

        # Compute features
        flows = torch.cat((forward_flow, backward_flow), dim=2)
        flows = flows.cpu()
        F = flow2F(flows.view(B * N, 2 * C, H, W)).view(B, N, 32, H, W)

        # Iterative Steps: todo: Should go to a function
        n_steps = 40
        for step in tqdm(range(n_steps), desc='## Step ##', position=0):
            confidence_new = current_confidence * 0.
            new_F = F * 0.
            for n_frame in tqdm(range(N), desc='  Frame', position=1, leave=False):
                if n_frame + 1 < N:
                    flow_from_features = F2flow(F[:, n_frame,:,:,:])


                    grid_f = flow_from_features[:, :2, :,:].permute((0, 2, 3, 1)) + ind

                    # Normalize the coordinates to the square [-1,1]
                    grid_f = (2 * grid_f / torch.tensor([W, H]).cpu().view(1, 1, 1, 2)) - 1

                    #warp ## FORWARD ##
                    F_f = torch.nn.functional.grid_sample(F[:, n_frame + 1,:,:,:], grid_f,
                                                          mode='bilinear', padding_mode='border', align_corners=False)
                    confidence_f = torch.clamp(
                         torch.nn.functional.grid_sample(current_confidence[:,n_frame + 1, :,:,:], grid_f, mode='bilinear',
                                                    padding_mode='border', align_corners=False), 0, 1)
                else:
                    F_f = 0. * F[:, n_frame]
                    confidence_f = 0. * current_confidence[:, n_frame]

                if n_frame -1 >= 0:

                    grid_b = flow_from_features[:, 2:].permute((0, 2, 3, 1)) + ind  # compute backward flow from features

                    # Normalize the coordinates to the square [-1,1]
                    grid_b = (2 * grid_b / torch.tensor([W, H]).cpu().view(1, 1, 1, 2)) - 1

                    # warp  ## BACKWARD ##
                    F_b = torch.nn.functional.grid_sample(F[:, n_frame - 1,:,:,:], grid_b, mode='bilinear', padding_mode='border',
                                                      align_corners=False)
                    confidence_b = torch.clamp(
                        torch.nn.functional.grid_sample(current_confidence[:, n_frame - 1,:,:,:], grid_b,
                                                        mode='bilinear',padding_mode='border', align_corners=False), 0, 1)
                else:
                    F_b = 0. * F[:, n_frame]
                    confidence_b = 0. * current_confidence[:, n_frame]
                #--

                # input of the update network is the concatenation of the obtained features from this frame and the neighboring ones
                x = torch.cat((F_b, F[:, n_frame], F_f), dim=1)
                confidence_in = torch.cat(((confidence_b).repeat(1, F.shape[2]//2, 1, 1),
                                     current_confidence[:, n_frame].repeat(1, F.shape[2]//2, 1, 1),
                                     (confidence_f).repeat(1, F.shape[2]//2, 1, 1)), dim=1)  # same goes for the input mask
                new_F[:, n_frame], confidence_new[:, n_frame] = update_net(x, confidence_in)  # Update
                # force the initially confident pixels to stay confident, because a decay can be observed
                # depending on the update rule of the partial convolution
                confidence_new[:, n_frame][hole[:, n_frame] == 1] = 1.

            F = new_F * 1.
            current_confidence = confidence_new * 1.  # mask update before next step

            completed_forward_flow = []
            completed_backward_flow = []
            for n_frame in range(N):
                with torch.no_grad():
                    inpainted_flow = F2flow(F[:, n_frame]).cpu().numpy()


                # Ver que hacer con esto --V--
                tmp_mask = confidence_new[:, n_frame].cpu().numpy() > 0
                tmp_mask = np.concatenate((tmp_mask, tmp_mask), axis=1)
                inpainted_flow = inpainted_flow * tmp_mask


                completed_forward_flow.append(inpainted_flow[0,:2].transpose(1, 2, 0))
                completed_backward_flow.append(inpainted_flow[0, 2:].transpose(1, 2, 0))



            if args.verbose:
                folder = join(args.verbose_path, 'inpainted_forward_flow', 'step%03d'%step)
                print('Saving inpainted forward flow into ' + folder)
                save_flow(completed_forward_flow, folder)

                folder = join(args.verbose_path, 'inpainted_backward_flow', 'step%03d'%step)
                print('Saving inpainted backward flow into ' + folder)
                save_flow(completed_backward_flow, folder)

        # end iterative steps

    return completed_forward_flow, completed_backward_flow



def main(args):

    assert args.mode in ('object_removal', 'video_extrapolation'), (
        "Accepted values for --mode: 'object_removal', 'video_extrapolation', but input is --mode %s"
    ) % args.mode

    if args.seamless:
        object_removal_seamless(args)
    else:
        video_completion(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--flow2features_weights', default=None, help='Path to the weights of the  encoder (flow to features) network architecture')
    parser.add_argument('--features2flow_weights', default=None, help='Path to the weights of the  decoder (features to flow) network architecture')
    parser.add_argument('--update_weights', default=None, help='Path to the weights of the update network architecture')
    parser.add_argument('--update_number_model', default='2', help='Class number of the update network architecture')
    parser.add_argument('--kind_update', default='pow', help='Class number of the update network architecture')

    parser.add_argument('--mode', default='object_removal', help="modes: object_removal / video_extrapolation")
    parser.add_argument('--seamless', action='store_true', help='Whether operate in the gradient domain')
    parser.add_argument('--video_path', default='../data/tennis', help="dataset for evaluation")
    parser.add_argument('--path_mask', default='../data/tennis_mask', help="mask for object removal")
    parser.add_argument('--outroot', default='../result/', help="output directory")

    parser.add_argument('--verbose', action='store_true', help='use small model')
    parser.add_argument('--verbose_path', default='../verbose_output', help='where intermediate results will be saved')


    # RAFT
    parser.add_argument('--opticalFlow_model', default='../weight/raft-things.pth', help="restore checkpoint for computing OF")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    args = parser.parse_args()
    main(args)