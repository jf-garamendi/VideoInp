import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

import streamlit as st
from pathlib import Path
from glob import glob

from os import listdir
from os.path import join


# TODO: Remove this, the parameters should be saved in the checkpoint and readed from it.
import training.training_parameters as training_param

def main():
    # Render the readme as markdown
    readme_text = st.markdown(read_markdown_file("../README.md"))

    st.sidebar.title('Choose Training')
    selected_train = choose_folder_inside_folder(folder = training_param.CHECKPOINT_ROOT_DIR, title ="Trainings list:")

    # TODO: Read the checkpoint

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

    selected_video = choose_folder_inside_folder(folder=data_folder, title="Available videos")

    # TODO: Read the frames
    st.sidebar.title("Frame")
    frame_idx, frame_name = frame_selector_ui(join(data_folder,selected_video))

    st.sidebar.title(frame_name)


    # TODO: show the frames

    # TODO: read the flow

    # TODO: show the flows

    # TODO: Read the masks

    # TODO: Show the masks

    # TODO: Create somewhere a function to inpaint a folder, Call it

    # TODO: show results

    # TODO: show diference between results and GT



def choose_folder_inside_folder(title ="", folder = None):

    inside_folder_list = listdir(folder)
    inside_folder_list = sorted(inside_folder_list)

    selected = st.sidebar.selectbox(title, inside_folder_list)

    return selected

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