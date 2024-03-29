"""
The Base Agent class, where all other agents inherit from, that contains definitions for all the necessary functions
"""
import logging


class BaseAgent:
    """
    This base class will contain the base functions to be overloaded by any agent you will implement.
    """

    def __init__(self):
        self.mode = None
        self.logger = logging.getLogger("Agent")

    def load_checkpoint(self, chk_config):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        raise NotImplementedError

    def save_checkpoint(self, chk_config, is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
        :return:
        """
        raise NotImplementedError

    
    def run(self):
        """
        The main operator
        :return:
        """
        assert self.mode in ['train', 'test']

        try:
            if self.mode == 'test':
                self.test()
            elif self.mode == 'train':
                self.train()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")
    
    
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

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        pass

    def test(self):
        pass