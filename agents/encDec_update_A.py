"""
The Base Agent class, where all other agents inherit from, that contains definitions for all the necessary functions
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

from agents.base import  BaseAgent
from os.path import join
from ingestion.VideoInp_DataSet import VideoInp_DataSet
from torch.utils.data import DataLoader
from torch import optim
from graphs.models import *
from utils.data_io import create_dir
class EncDec_update_agent_A(BaseAgent):
    """
    This base class will contain the base functions to be overloaded by any agent you will implement.
    """

    def __init__(self,  config):
        super().__init__()

        ######################
        # Folders
        self.exp_name = config.general.exp_name
        self.tb_stats_dir = join(config.verbose.tensorboard_root_dir, self.exp_name)
        self.checkpoint_dir = join(config.checkpoint.root_dir, self.exp_name)

        #Datasets
        self.set_data(config.data)

        # Models
        self.set_model(config.model)

        # Optimizers
        self.set_optimizer(config.optimizer)

        #################################
        # If exists checkpoints, load them
        self.load_checkpoint(config.checkpoint.root_dir)

        AQUI ME HE QUEDADO


    ########################################################################
    # SET FUNCTIONS
    def set_data(self, data_config):
        train_data = VideoInp_DataSet(data_config.train_root_dir,
                                      GT=True,
                                      number_of_frames = data_config.number_of_frames,
                                      random_holes_on_the_fly = data_config.data.random_holes_on_the_fly,
                                      n_random_holes_per_frame = data_config.data.n_random_holes_per_frame
                                      )

        self.train_loader = DataLoader(train_data, batch_size=1, shuffle=True, drop_last=False)

        val_data = VideoInp_DataSet(data_config.val_root_dir,
                                    number_of_frames=data_config.data.number_of_frames,
                                    GT=True,
                                    random_holes_on_the_fly=False)
        self.val_loader = DataLoader(val_data, batch_size=1, shuffle=False, drop_last=False)

    def set_model(self, model_config):
        self.encoder_decoder = globals()[model_config.encDec]
        self.encoder_decoder = self.encoder_decoder()

        self.update = globals()[model_config.update]
        self.update = self.update(update=model_config.partial_mode_update)

    def set_optimizer(self, optim_config):
        self.encDec_optimizer = optim.Adam(self.encoder_decoder.parameters(),
                                           lr=optim_config.learning_rate,
                                           betas=(optim_config.beta_1, optim_config.beta_2),
                                           weight_decay=optim_config.weight_decay)

        self.update_optimizer = optim.Adam(self.update.parameters(),
                                           lr=optim_config.learning_rate,
                                           betas=(optim_config.beta_1, optim_config.beta_2),
                                           weight_decay=optim_config.weight_decay)

    # END of SET FUNCTIONS
    #######################################################################


    def load_checkpoint(self, chk_config):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        create_dir(chk_config.root_dir)
        self.encDec_checkpoint_filename = join(chk_config.root_dir, chk_config.enc_dec_checkpoint_filename)
        self.encoder_decoder.load_checkpoint(self.encDec_checkpoint_filename)

        self.update_checkpoint_filename = join(chk_config.root_dir, chk_config.update_checkpoint_filename)
        self.update.load_checkpoint(self.update_checkpoint_filename)

    def save_checkpoint(self, chk_config, is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
        :return:
        """
        raise NotImplementedError

    def train_one_epoch(self):
        """
            One epoch of training
            :return:
        """
        raise NotImplementedError

    def train(self):
        """
        Main training loop
        :return:
        """
        raise NotImplementedError

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        raise NotImplementedError

    def inference(self):
        """
        Make inference for one data element

        :return:
        """
        raise NotImplementedError