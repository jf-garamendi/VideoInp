# Name of the training
#TRAINING_NAME = "Pierrick_Complete_001"
TRAINING_NAME = "conv3D_debug"

###########
# DATASET
###########
encdDec_random_mask_on_the_fly = False
update_random_mask_on_the_fly = True
n_masks = 5

# where the training dataset is located
#ENC_DEC_TRAIN_ROOT_DIR = '../../../data/datasets/built/davis_no_mask'
ENC_DEC_TRAIN_ROOT_DIR = '../../../data/datasets/built/5Tennis_b'
ENC_DEC_TEST_ROOT_DIR = '../../../data/datasets/built/5Tennis_b'


UPDATE_TRAIN_ROOT_DIR = '../../../data/datasets/built/5Tennis_b'
#UPDATE_TRAIN_ROOT_DIR = '../../../data/datasets/built/davis_no_mask'
UPDATE_TEST_ROOT_DIR = '../../../data/datasets/built/5Tennis_b'

#####
# Training status visualization
####
# The loss is shown, and the images are saved  every SHOWING_N_ITER
SHOWING_N_ITER = 1
# The weights are saved every SAVING_N_ITER
SAVING_N_ITER = 1

#######################################################################################################
# TRAINING PARAMETERS
#######################################################################################################
encDec_n_epochs = 1100
update_n_epochs = 30000

max_num_steps = 20
ingestion_number_of_frames = 10

## For adam optimizer
adam_lr = 1e-4
adam_betas = (0.9, 0.999)
adam_weight_decay = 4e-5

# Losses. Each loss has to have the corresponding weight. The final loss will be the summ of all losses
# multiplied by its corresponding weight
import model.losses as loss
encDec_losses = {
    'losses_list':  [loss.L1],
    'weights_list': [1]
           }

update_losses = {
    'losses_list':  [loss.L1, loss.TV, loss.minfbbf],
    'weights_list': [1, 10, 2]
           }

#partial Convolution

#Possible values:
#   partial_mode_update = 'pow'
#   partial_mode_update = 'pol'
partial_mode_update = 'pow'

# Tensor Board Root dir
TB_STATS_ROOT_DIR = '../../../data/verbose/tensor_board/'

# Output verbose images root dir
VERBOSE_ROOT_DIR = '../../../data/verbose/training_out/'

# weights root dir and file name
CHECKPOINT_ROOT_DIR = '../../../data/verbose/checkpoint/'
ENC_DEC_CHECKPOINT_FILENAME = 'enc_dec_checkpoint.tar'
UPDATE_CHECKPOINT_FILENAME = 'update_checkpoint.tar'

#######################################################################################################
# Next part of the file  contains all constant independent of model, hyper parameters, parameters, etc.
# Usually, you do not need to change the following variables
#######################################################################################################


# name of the folders inside TRAIN_ROOT_DIR and TEST_ROOT_DIR

#folder where the frames are in the raw dataset (for each sequence). This folder is used by create_dataset.py
#RAW_FRAMES_FOLDER = "frames" #<-- For the example in README.md
RAW_FRAMES_FOLDER = "." # <-- For DAVIS

#folder where the training (validation) frames are . These folders is generated by create_dataset.py and used for training
MASKS_FOLDER = "masks"
FRAMES_FOLDER = "frames"

GT_FRAMES_FOLDER = "gt_frames"

FWD_FLOW_FOLDER = "fwd_flow"
BWD_FLOW_FOLDER = "bwd_flow"

GT_FWD_FLOW_FOLDER = "gt_fwd_flow"
GT_BWD_FLOW_FOLDER = "gt_bwd_flow"

