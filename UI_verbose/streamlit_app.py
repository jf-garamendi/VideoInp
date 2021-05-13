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
from utils.data_io import read_frame, read_flow
from utils.flow_viz import flow_to_image

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


    # From the available trainings, choose one
    st.sidebar.title('Choose Training')
    verbose_root_dir = training_param.VERBOSE_ROOT_DIR
    _, selected_train = choose_folder_inside_folder(folder = verbose_root_dir, title ="Trainings list:")

    which_part = st.sidebar.radio("What model component do you want to see?", ("Encoder-Decoder", "Update"), index=1)
    fwd_or_bck = st.sidebar.radio("What flow do you want to see?", ("Forward", "Backward"), index=0)

    #Read each folder
    folder_train = join(verbose_root_dir, selected_train)
    folders = sorted(listdir(folder_train))
    in_flow_names = []
    in_flow_imgs = []
    txt_for_in_flow =""
    txt_for_comp_flow = ""
    comp_flow_imgs = []
    gt_flow_imgs = []
    for folder in folders:
        #st.text(folder)
        if ("encDec" in folder) and (which_part == "Encoder-Decoder"):
            if ("input" in folder):
                #st.header("Input Optical Flow")
                    if ("forward" in folder) and (fwd_or_bck == "Forward"):
                        in_flow_names, in_flow_imgs = read_flows_from_folder(join(folder_train, folder))
                        txt_for_in_flow = "Input Forward Flow"
                    elif ("backward" in folder) and (fwd_or_bck == "Backward"):
                        in_flow_names, in_flow_imgs = read_flows_from_folder(join(folder_train, folder))
                        txt_for_in_flow = "Input BackwardFlow"

            if ('computed' in folder):
                if ("forward" in folder) and (fwd_or_bck == "Forward"):
                    comp_flow_names, comp_flow_imgs = read_flows_from_folder(join(folder_train, folder))
                elif ("backward" in folder) and (fwd_or_bck == "Backward"):
                    comp_flow_names, comp_flow_imgs = read_flows_from_folder(join(folder_train, folder))

            if ('GT' in folder):
                if ("forward" in folder) and (fwd_or_bck == "Forward"):
                    gt_flow_names, gt_flow_imgs = read_flows_from_folder(join(folder_train, folder))
                elif ("backward" in folder) and (fwd_or_bck == "Backward"):
                    gt_flow_names, gt_flow_imgs = read_flows_from_folder(join(folder_train, folder))


        if ("pdate" in folder) and (which_part == "Update"):
            if ("input" in folder):
                #st.header("Input Optical Flow")
                    if ("forward" in folder) and (fwd_or_bck == "Forward"):
                        in_flow_names, in_flow_imgs = read_flows_from_folder(join(folder_train, folder))
                        txt_for_in_flow = "Input Forward Flow"
                    elif ("backward" in folder) and (fwd_or_bck == "Backward"):
                        in_flow_names, in_flow_imgs = read_flows_from_folder(join(folder_train, folder))


            if ('computed' in folder):
                if ("forward" in folder) and (fwd_or_bck == "Forward"):
                    comp_flow_names, comp_flow_imgs = read_flows_from_folder(join(folder_train, folder))
                elif ("backward" in folder) and (fwd_or_bck == "Backward"):
                    comp_flow_names, comp_flow_imgs = read_flows_from_folder(join(folder_train, folder))

            if ('GT' in folder):
                if ("forward" in folder) and (fwd_or_bck == "Forward"):
                    gt_flow_names, gt_flow_imgs = read_flows_from_folder(join(folder_train, folder))
                elif ("backward" in folder) and (fwd_or_bck == "Backward"):
                    gt_flow_names, gt_flow_imgs = read_flows_from_folder(join(folder_train, folder))


    st.title(which_part)

    if len(in_flow_imgs)>0:
        selected_frame_index = st.sidebar.slider("Choose a frame (index)", 0, len(in_flow_imgs) - 1, 1)

        if len(gt_flow_imgs)>0:
            img_flow_diff = flow_to_image(gt_flow_imgs[selected_frame_index] - comp_flow_imgs[selected_frame_index])
        else:
            img_flow_diff = flow_to_image(0*comp_flow_imgs[selected_frame_index])

        img_in_flow = flow_to_image(in_flow_imgs[selected_frame_index])
        img_comp_flow = flow_to_image(comp_flow_imgs[selected_frame_index])
        img_gt_flow = flow_to_image(gt_flow_imgs[selected_frame_index])

        st.image([img_in_flow, img_gt_flow, img_comp_flow , img_flow_diff], ["Input", "Ground Truth","Computed",  "Error"])

    else:
        st.header("There are no data " )




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

def read_flows_from_folder(folder):
    '''
    folder = join(folder, "flow_png")

    files = sorted(listdir(folder))

    flows = []
    for file in files:
        flows.append(read_frame(join(folder, file)))

    return files, flows
    '''

    folder = join(folder, "flow_flo")

    files = sorted(listdir(folder))

    flows = []
    for file in files:
        flows.append(read_flow(join(folder, file)))

    return files, flows
if __name__ == "__main__":
    main()