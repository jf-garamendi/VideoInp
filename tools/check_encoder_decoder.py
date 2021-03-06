import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
import argparse
import torch
from os.path import join

from tqdm import tqdm

from video_completion import calculate_flow, load_video_frames, initialize_RAFT, initialize_encoder, initialize_decoder

from utils import flow_viz, frame_utils
from PIL import Image
import numpy as np

def main(args):

    #read the frames
    video = load_video_frames(args.video_path)
    video = video.to('cuda')

    #Flow model
    RAFT_model = initialize_RAFT(args)

    # Calcutes the flow. Notice that this flow is computed  on the non-masked video
    forward_flow = calculate_flow(RAFT_model, video, 'forward')
    # add the lid to the final
    # forward_flow = np.concatenate((forward_flow, np.zeros((imgH, imgW, 2,1))), axis=3)
    backward_flow = calculate_flow(RAFT_model, video, 'backward')
    # add the lid to the beginning
    # backward_flow = np.concatenate((np.zeros((imgH, imgW, 2,1)), backward_flow), axis=3)

    # re-order into the Pierrick's ordering
    forward_flow = np.expand_dims(np.stack(forward_flow, axis=3), axis=0)
    backward_flow = np.expand_dims(np.stack(backward_flow, axis=3), axis=0)

    forward_flow = torch.from_numpy(forward_flow).permute(0, 4, 3, 1, 2).contiguous().float()
    backward_flow = torch.from_numpy(backward_flow).permute(0, 4, 3, 1, 2).contiguous().float()

    forward_flow.cpu()
    backward_flow.cpu()

    flows = torch.cat((forward_flow, backward_flow), dim=2)
    flows = flows.cpu()

    B, N, C, H, W = forward_flow.shape


    with torch.no_grad():
        ###
        ### Build the complete model and Load the weights

        # encoder Flow to Features
        flow2F = initialize_encoder(args.flow2features_weights)

        # decoder Features to Flow
        F2flow = initialize_decoder(args.features2flow_weights)

        # Compute features
        F = flow2F(flows.view(B * N, 2 * C, H, W)).view(B, N, 32, H, W)


        if args.features_perturbation:
            perturbation = np.random.random(F.shape).astype(np.float32)       # Random perturbation
        else:
            perturbation = np.ones(F.shape).astype(np.float32)  # No perturbation

        F = F*torch.from_numpy(perturbation)

        #Test decode
        for n_frame in tqdm(range(N), desc='# Frame', position=1, leave=False):

            flow_from_features = F2flow(F[:, n_frame, :, :, :]).cpu().numpy()
            flow_GT = flows.numpy()[:, n_frame, :, :, :]

            # Flow visualization.
            NO_fwd_flow = flow_from_features[0, :2].transpose(1, 2, 0)
            NO_fwd_flow_img = flow_viz.flow_to_image(NO_fwd_flow)
            NO_fwd_flow_img = Image.fromarray(NO_fwd_flow_img)

            NO_bwd_flow = flow_from_features[0, 2:].transpose(1, 2, 0)
            NO_bwd_flow_img = flow_viz.flow_to_image(NO_bwd_flow)
            NO_bwd_flow_img = Image.fromarray(NO_bwd_flow_img)

            GT_fwd_flow = flow_GT[0, :2].transpose(1, 2, 0)
            GT_fwd_flow_img = flow_viz.flow_to_image(GT_fwd_flow)
            GT_fwd_flow_img = Image.fromarray(GT_fwd_flow_img)

            GT_bwd_flow = flow_GT[0, 2:].transpose(1, 2, 0)
            GT_bwd_flow_img = flow_viz.flow_to_image(GT_bwd_flow)
            GT_bwd_flow_img = Image.fromarray(GT_bwd_flow_img)

            # Saves the flow and flow_img.
            dir = join(args.saving_path, 'forward_flow')
            if not os.path.exists(dir):
                os.makedirs(dir)

            GT_png_name = dir + '/GT_%05d.png' % n_frame
            NO_png_name = dir + '/NO_%05d.png' % n_frame

            GT_fwd_flow_img.save(GT_png_name)
            NO_fwd_flow_img.save(NO_png_name)


            dir = os.path.join(args.saving_path, 'backward_flow')
            if not os.path.exists(dir):
                os.makedirs(dir)
            GT_png_name = dir + '/GT_%05d.png' % n_frame
            NO_png_name = dir + '/NO_%05d.png' % n_frame
            GT_bwd_flow_img.save(GT_png_name)
            NO_bwd_flow_img.save(NO_png_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--saving_path',
                        help='Root folder where the results will be saved')

    parser.add_argument('--flow2features_weights', default=None,
                        help='Path to the weights of the  encoder (flow to features) network architecture')
    parser.add_argument('--features2flow_weights', default=None,
                        help='Path to the weights of the  decoder (features to flow) network architecture')

    parser.add_argument('--video_path', default='../data/tennis', help="dataset for evaluation")

    parser.add_argument('--features_perturbation', action='store_true', help='add random noise to features')

    # RAFT
    parser.add_argument('--opticalFlow_model', default='../weight/raft-things.pth',
                        help="restore checkpoint for computing OF")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')



    args = parser.parse_args()

    main(args)