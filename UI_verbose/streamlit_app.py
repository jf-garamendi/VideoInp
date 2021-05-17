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

def main(verbose_root_dir):
    ####################

    # Render the readme as markdown
    # readme_text = st.markdown(read_markdown_file("../README.md"))


    # From the available trainings, choose one
    st.sidebar.title('Choose Training A')
    _, selected_train_A = choose_folder_inside_folder(folder = verbose_root_dir, title ="Trainings list A:")

    st.sidebar.title('Choose Training B (to compare)')
    _, selected_train_B = choose_folder_inside_folder(folder=verbose_root_dir, title="Trainings list B:")

    which_part = st.sidebar.radio("What model component do you want to see?", ("Encoder-Decoder", "Update"), index=1)
    fwd_or_bck = st.sidebar.radio("What flow do you want to see?", ("Forward", "Backward"), index=0)
    A_or_B = st.sidebar.radio("Which model do you want to see?", (selected_train_A, selected_train_B), index=0)
    overlay_GT_on_comp = st.sidebar.checkbox("Replace computed OF by the GT", value=False)

    #Read each folder
    folder_train_A = join(verbose_root_dir, selected_train_A)
    folder_train_B = join(verbose_root_dir, selected_train_B)


    in_flow_A, gt_flow_A, comp_flow_A  = read_flows_from_foldertrain(folder_train_A, which_part, fwd_or_bck)
    in_flow_B, gt_flow_B, comp_flow_B = read_flows_from_foldertrain(folder_train_B, which_part, fwd_or_bck)

    if A_or_B == selected_train_A:
        in_flow, gt_flow, comp_flow = in_flow_A, gt_flow_A, comp_flow_A
    else:
        in_flow, gt_flow, comp_flow = in_flow_B, gt_flow_B, comp_flow_B


    if len(in_flow)>0:
        selected_frame_index = st.sidebar.slider("Choose a frame (index)", 0, len(in_flow) - 1, 1)


        if len(gt_flow)>0:
            img_flow_diff = flow_to_image(gt_flow[selected_frame_index] - comp_flow[selected_frame_index])
        else:
            img_flow_diff = flow_to_image(0*comp_flow[selected_frame_index])

        img_in_flow = flow_to_image(in_flow[selected_frame_index])
        img_gt_flow = flow_to_image(gt_flow[selected_frame_index])
        if overlay_GT_on_comp:
            img_comp_flow = img_gt_flow
        else:
            img_comp_flow = flow_to_image(comp_flow[selected_frame_index])

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

@st.cache(suppress_st_warning=True)
def read_flows_from_folder(folder):
    folder = join(folder, "flow_flo")

    files = sorted(listdir(folder))

    flows = []
    for file in files:
        flows.append(read_flow(join(folder, file)))

    return files, flows

@st.cache(suppress_st_warning=True)
def read_flows_from_foldertrain(folder_train, which_model_part, which_flow):
    #Read each folder
    folders = sorted(listdir(folder_train))
    in_flow = []
    comp_flow = []
    gt_flow = []
    for folder in folders:
        #st.text(folder)
        if ("encDec" in folder) and (which_model_part == "Encoder-Decoder"):
            if ("input" in folder):
                    if ("forward" in folder) and (which_flow == "Forward"):
                        _, in_flow = read_flows_from_folder(join(folder_train, folder))
                    elif ("backward" in folder) and (which_flow == "Backward"):
                        _, in_flow = read_flows_from_folder(join(folder_train, folder))


            if ('computed' in folder):
                if ("forward" in folder) and (which_flow == "Forward"):
                    _, comp_flow = read_flows_from_folder(join(folder_train, folder))
                elif ("backward" in folder) and (which_flow == "Backward"):
                    _, comp_flow = read_flows_from_folder(join(folder_train, folder))

            if ('GT' in folder):
                if ("forward" in folder) and (which_flow == "Forward"):
                    _, gt_flow = read_flows_from_folder(join(folder_train, folder))
                elif ("backward" in folder) and (which_flow == "Backward"):
                    _, gt_flow = read_flows_from_folder(join(folder_train, folder))


        if ("pdate" in folder) and (which_model_part == "Update"):
            if ("input" in folder):
                    if ("forward" in folder) and (which_flow == "Forward"):
                        _, in_flow = read_flows_from_folder(join(folder_train, folder))
                    elif ("backward" in folder) and (which_flow == "Backward"):
                        _, in_flow = read_flows_from_folder(join(folder_train, folder))


            if ('computed' in folder):
                if ("forward" in folder) and (which_flow == "Forward"):
                    _, comp_flow = read_flows_from_folder(join(folder_train, folder))
                elif ("backward" in folder) and (which_flow == "Backward"):
                    _, comp_flow = read_flows_from_folder(join(folder_train, folder))

            if ('GT' in folder):
                if ("forward" in folder) and (which_flow == "Forward"):
                    _, gt_flow = read_flows_from_folder(join(folder_train, folder))
                elif ("backward" in folder) and (which_flow == "Backward"):
                    _, gt_flow = read_flows_from_folder(join(folder_train, folder))


    return in_flow, gt_flow, comp_flow

if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder = sys.argv[1]
        print("Reading data from: ", folder)
        main(folder)
    else:
        print("Please  specify the folder to read as first argument")
        st.stop()
