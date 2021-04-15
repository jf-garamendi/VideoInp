# Name of the training
TRAINING_NAME = "Pierrick_Overfit_012"

###########
# DATASET
###########
encdDec_random_mask_on_the_fly = False
update_random_mask_on_the_fly = False

# where the training dataset is located
ENC_DEC_TRAIN_ROOT_DIR = '../datasets/5Tennis_no_mask'
ENC_DEC_TEST_ROOT_DIR = '../datasets/5Tennis_b'


UPDATE_TRAIN_ROOT_DIR = '../datasets/5Tennis_b'
UPDATE_TEST_ROOT_DIR = '../datasets/5Tennis_b'

#####
# Training status visualization
####
# The loss is shown, and the images are saved  every SHOWING_N_ITER
SHOWING_N_ITER = 5
# The weights are saved every SAVING_N_ITER
SAVING_N_ITER = 5

#######################################################################################################
# TRAINING PARAMETERS
#######################################################################################################
encDec_n_epochs = 3000
update_n_epochs = 300000

max_num_steps = 20

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
    'losses_list':  [loss.L1],
    'weights_list': [1]
           }

#partial Convolution

#Possible values:
#   partial_mode_update = 'pow'
#   partial_mode_update = 'pol'
partial_mode_update = 'pow'

# Tensor Board Root dir
TB_STATS_ROOT_DIR = './tensor_board/'

# Output verbose images root dir
VERBOSE_ROOT_DIR = './training_out/'

# weights root dir and file name
CHECKPOINT_ROOT_DIR = './checkpoint/'
ENC_DEC_CHECKPOINT_FILENAME = 'enc_dec_checkpoint.tar'
UPDATE_CHECKPOINT_FILENAME = 'update_checkpoint.tar'

#######################################################################################################
# Next part of the file  contains all constant independent of model, hyper parameters, parameters, etc.
# Usually, you do not need to change the following variables
#######################################################################################################


# name of the folders inside TRAIN_ROOT_DIR and TEST_ROOT_DIR
FRAMES_FOLDER = "frames"
MASKS_FOLDER = "masks"

GT_FRAMES_FOLDER = "gt_frames"

FWD_FLOW_FOLDER = "fwd_flow"
BWD_FLOW_FOLDER = "bwd_flow"

GT_FWD_FLOW_FOLDER = "gt_fwd_flow"
GT_BWD_FLOW_FOLDER = "gt_bwd_flow"

