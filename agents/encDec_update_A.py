"""
The Base Agent class, where all other agents inherit from, that contains definitions for all the necessary functions
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
import logging
from agents.base import  BaseAgent
from os.path import join
from training.VideoInp_DataSet import VideoInp_DataSet

class EncDec_update_agent_A(BaseAgent):
    """
    This base class will contain the base functions to be overloaded by any agent you will implement.
    """

    def __init__(self,  config, **models):
        super().__init__(config)

        # Folders
        self.exp_name = config.general.exp_name
        self.TB_stats_dir = join(config.verbose.tensor_board_root_dir, self.exp_name)
        self.checkpoint_dir = join(config.checkpoint.root_dir, self.exp_name)

        #Datasets
        encDec_train_data = VideoInp_Dataset(config.data.train_root_dir,
                                             GT = True,
                                             number_of_frames = config.data.number_of_frames,
           AQUI ME HE QUEDADO

        )





    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        raise NotImplementedError

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
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