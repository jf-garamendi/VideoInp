import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

import streamlit as st
from pathlib import Path
from glob import glob

from os import listdir
from os.path import join

from training.VideoInp_DataSet import VideoInp_DataSet
from torch.utils.data import DataLoader
from training.launch_training import update_step


# TODO: Remove this, the parameters should be saved in the checkpoint and readed from it.
import training.training_parameters as training_param

from PIL import Image
import numpy as np
from utils.flow_viz import flow_to_image
import torch
from model.iterative import Flow2features, Features2flow, Res_Update

def main():
    ####################

    # Render the readme as markdown
    # readme_text = st.markdown(read_markdown_file("../README.md"))

    st.sidebar.title('Choose Training')
    _, selected_train = choose_folder_inside_folder(folder = training_param.CHECKPOINT_ROOT_DIR, title ="Trainings list:")

    # Read the checkpoint
    # Net Models
    flow2F = Flow2features()
    F2flow = Features2flow()
    update_net = Res_Update(update=training_param.partial_mode_update)
    encDec_checkpoint_filename = join(training_param.CHECKPOINT_ROOT_DIR, selected_train,
                                      training_param.ENC_DEC_CHECKPOINT_FILENAME)
    update_checkpoint_filename = join(training_param.CHECKPOINT_ROOT_DIR, selected_train,
                                      training_param.UPDATE_CHECKPOINT_FILENAME)

    checkpoint = torch.load(encDec_checkpoint_filename, map_location='cpu')

    flow2F.load_state_dict(checkpoint['enc_state_dict'])
    F2flow.load_state_dict(checkpoint['dec_state_dict'])

    checkpoint = torch.load(update_checkpoint_filename, map_location='cpu')
    update_net.load_state_dict(checkpoint['update_state_dict'])

    ##############
    st.sidebar.title('Choose which video to see')
    data_type = st.sidebar.radio("Choose the type of dataset", ["Training", "Validation", "Testing"])

    data_folder = '.'
    if data_type == "Training":
        data_folder = training_param.TRAIN_DATA_ROOT_DIR
    elif data_type == "Validation":
        data_folder = training_param.VAL_DATA_ROOT_DIR
    elif data_type == "Testing":
        data_folder = training_param.TEST_DATA_ROOT_DIR

    #################
    random_mask = st.sidebar.checkbox("Random mask")
    # Create the tools for reading
    data = VideoInp_DataSet(data_folder,
                                         training=True,
                                         number_of_frames = -1,
                                         random_mask_on_the_fly = random_mask,
                                         n_masks=training_param.n_masks)


    video_idx, selected_video = choose_folder_inside_folder(folder=data_folder, title="Available videos")

    frames, flows, masks, gt_frames, gt_flows = data.__getitem__(video_idx)

    # Select frame number
    st.sidebar.title("Frame Number")
    frame_idx, frame_name = frame_selector_ui(join(data_folder, selected_video))

    st.sidebar.text(frame_name)

    # show the frame
    frame = frames[frame_idx, :, :, :].permute(1,2,0)
    img_frame = (255 * np.squeeze(frame.numpy())).astype(np.uint8)

    st.image(img_frame, use_column_width=True)

    # show the flows
    flow_frame = flows[frame_idx, 0:2, :, :].permute((1, 2, 0))
    flow_frame = np.squeeze(flow_frame.numpy())

    flow_img = flow_to_image(flow_frame)

    st.image(flow_img, use_column_width=True)

    # Show the masks
    mask_frame = 255* np.squeeze(masks[frame_idx,:,:].numpy()).astype((np.uint8))
    st.image(mask_frame)


    # TODO: Inpaint
    N, C, H, W = flows.shape
    B=1


    frames = frames.view(B * N, 3, H, W)
    flows = flows.view(B * N, C, H, W)
    # masks: 1 inside the hole
    masks = masks.view(B * N, 1, H, W)
    gt_frames = gt_frames.view(B * N, 3, H, W)
    gt_flows = gt_flows.view(B * N, C, H, W)

    # place data on device
    flows = flows.to('cpu')
    masks = masks.to('cpu')
    gt_flows = gt_flows.to('cpu')

    # Initial confidence: 1 outside the mask (the hole), 0 inside
    initial_confidence = 1 - 1. * masks
    confidence = initial_confidence.clone()
    gained_confidence = initial_confidence

    new_flow = flows.clone()
    F = flow2F(flows)

    step = -1

    while (gained_confidence.sum() > 0) and (step <= training_param.max_num_steps):
        step += 1
        # print('Test step: ', str(step))

        current_flow = new_flow.clone().detach()
        current_F = F.clone().detach()

        new_F, confidence_new = update_step(update_net, current_flow, current_F, confidence)
        gained_confidence = (confidence_new > confidence) * confidence_new

        if gained_confidence.sum() > 0:
            F = current_F * (confidence_new <= confidence) + new_F * (confidence_new > confidence)
            new_flow = F2flow(F)


        # mask update before next step
        confidence = confidence_new.clone().detach()


    # TODO: show results
    flow_frame = new_flow[frame_idx, 0:2, :, :].permute((1, 2, 0))
    flow_frame = np.squeeze(flow_frame.numpy())

    flow_img = flow_to_image(flow_frame)

    st.image(flow_img, use_column_width=True)


    # TODO: show diference between results and GT



def choose_folder_inside_folder(title ="", folder = None):

    inside_folder_list = listdir(folder)
    inside_folder_list = sorted(inside_folder_list)

    selected = st.sidebar.selectbox(title, inside_folder_list)
    idx = inside_folder_list.index(selected)

    return idx, selected

def frame_selector_ui(video_folder):
    frames_path = join(video_folder, training_param.FRAMES_FOLDER)
    frame_names = sorted(listdir(frames_path))

    # Choose a frame out of the selected frames.
    selected_item_index = st.sidebar.slider("Choose a frame (index)", min_value=0, max_value=len(frame_names)-1, value=1)

    #TODO: Draw an altair chart in the sidebar with information about the error. Take a look to the self driving example

    selected_item = frame_names[selected_item_index]

    return selected_item_index, selected_item


def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

if __name__ == "__main__":
    main()