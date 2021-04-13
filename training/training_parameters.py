# Name of the training
TRAINING_NAME = "Pierrick_Overfit_010"


# Training status visualization

# The loss is send, and the images are saved shown every SHOWING_N_ITER
SHOWING_N_ITER = 5
SAVING_N_ITER = 5

# --- v
VERBOSE_DIR ='../training_out/Pierrick_Overfit_010'
# ---- ^
#######################################################################################################
# TRAINING PARAMETERS
#######################################################################################################
n_epochs = 300000

S_0 = 1000

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
    'weights_list': [1, 1, 1]
           }

###########
# DATASET
###########
random_mask_on_the_fly = False

# where the training dataset is located
#TRAIN_ROOT_DIR = '../datasets/5Tennis_no_mask'
TRAIN_ROOT_DIR = '../datasets/5Tennis_b'

# where the testing dataset is located
TEST_ROOT_DIR = '../datasets/5Tennis_b'



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

# Tensor Board Root dir
TB_STATS_ROOT_DIR = './tensor_board/'

# Output verbose images root dir
VERBOSE_ROOT_DIR = './training_out/'

# weights root dir and file name
CHECKPOINT_ROOT_DIR = './checkpoint/'
CHECKPOINT_FILENAME = 'all.tar'