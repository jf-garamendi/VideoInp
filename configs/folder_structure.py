'''
This file contains all constant independent of model, hyper parameters, parameters, etc.
Usually, you do not need to change the following variables

'''




# name of the folders inside TRAIN_ROOT_DIR and VAL_ROOT_DIR

#folder where the frames are in the raw dataset (for each sequence). This folder is used by create_dataset.py
RAW_FRAMES_FOLDER = "frames" #<-- For the example in README.md
#RAW_FRAMES_FOLDER = "." # <-- For DAVIS TODO: Incluir una opcion DAVIS en create_dataset.py y eliminar este parametro de aui
RAW_MASKS_FOLDER = "masks"

#folder where the training (validation) frames are . These folders is generated by create_dataset.py and used for training
MASKS_FOLDER = "masks"
FRAMES_FOLDER = "frames"

GT_FRAMES_FOLDER = "gt_frames"

FWD_FLOW_FOLDER = "fwd_flow"
BWD_FLOW_FOLDER = "bwd_flow"

GT_FWD_FLOW_FOLDER = "gt_fwd_flow"
GT_BWD_FLOW_FOLDER = "gt_bwd_flow"