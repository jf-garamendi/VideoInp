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
import torch

from graphs.models import *
from graphs.losses import *

from utils.data_io import create_dir, verbose_images
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class EncDec_update_agent_A(BaseAgent):
    """
    This base class will contain the base functions to be overloaded by any agent you will implement.
    """

    def __init__(self,  config):
        super().__init__()

        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and config.general.device == "cuda") else "cpu")

        self.encDec_epoch = 0
        self.update_epoch = 0

        self.encDec_n_epochs = config.training.encDec_n_epochs
        self.update_n_epochs = config.training.update_n_epochs

        self.encDec_loss = None


        self.mode = config.general.mode

        ######################
        # Folders
        self.exp_name = config.general.exp_name
        self.tb_stats_dir = join(config.verbose.tensorboard_root_dir, self.exp_name)
        self.checkpoint_dir = join(config.checkpoint.root_dir, self.exp_name)
        self.val_out_images = config.verbose.val_out_images

        # Datasets
        self.set_data(config.data)

        # Models
        self.set_model(config.model)

        # Optimizers
        self.set_optimizer(config.optimizer)

        # Losses
        self.set_losses(config.losses)

        # Tensor Board writers
        self.TB_writer = SummaryWriter(config.verbose.tensorboard_root_dir)

        #################################
        # If exists checkpoints, load them
        self.load_checkpoint(config.checkpoint)


    ########################################################################
    # SET FUNCTIONS
    def set_data(self, data_config):
        train_data = VideoInp_DataSet(data_config.train_root_dir,
                                      GT=True,
                                      number_of_frames = data_config.number_of_frames,
                                      random_holes_on_the_fly = data_config.random_holes_on_the_fly,
                                      n_random_holes_per_frame = data_config.n_random_holes_per_frame
                                      )

        self.train_loader = DataLoader(train_data, batch_size=1, shuffle=True, drop_last=False)

        val_data = VideoInp_DataSet(data_config.val_root_dir,
                                    number_of_frames=data_config.number_of_frames,
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

    def set_losses(self, loss_config):

        self.encDec_losses_fn = []
        self.encDec_losses_weight = []

        self.update_losses_fn = []
        self.update_losses_weight = []

        for loss, weight in zip(loss_config.encDec_losses.losses, loss_config.encDec_losses.weights):
            self.encDec_losses_fn.append(globals()[loss]())
            self.encDec_losses_weight.append(weight)

        for loss, weight in zip(loss_config.update_losses.losses, loss_config.update_losses.weights):
            self.update_losses_fn.append(globals()[loss]())
            self.update_losses_weight.append(weight)

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


        if os.path.exists(self.encDec_checkpoint_filename):
            checkpoint = torch.load(self.encDec_checkpoint_filename, map_location='cpu')

            self.encoder_decoder.load_state_dict(checkpoint['encDec_state_dict'])
            self.encDec_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.encDec_epoch = checkpoint['epoch']

            print('** Checkpoint ' + self.encDec_checkpoint_filename + ' loaded \n')


        self.update_checkpoint_filename = join(chk_config.root_dir, chk_config.update_checkpoint_filename)

        if os.path.exists(self.update_checkpoint_filename):
            checkpoint = torch.load(self.update_checkpoint_filename, map_location='cpu')
            self.update.load_state_dict(checkpoint['update_state_dict'])
            self.update_optimizer.load_state_dict((checkpoint['optimizer_state_dict']))
            self.update_epoch = checkpoint['epoch']

            print('** Checkpoint ' + self.update_checkpoint_filename + ' loaded \n')

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

    def train_one_epoch_encDec(self):
        # train mode
        self.encoder_decoder.train()

        loss2print = [0] * len(self.encDec_losses_fn)

        for data in tqdm(self.train_loader, leave=False, desc='    Videos: '):
            _, flows, masks, _, gt_flows = data

            B, N, C, H, W = flows.shape

            # Remove the batch dimension (for pierrick architecture is needed B to be 1)
            flows = flows.view(B * N, C, H, W)
            gt_flows = gt_flows.view(B * N, C, H, W)

            self.encDec_optimizer.zero_grad()
            # Forward Pass
            computed_flows = self.encoder_decoder(flows)

            # Compute loss
            train_loss = torch.tensor(0).to(self.device)
            i = 0
            for loss, weight in zip(self.encDec_losses_fn, self.encDec_losses_weight):
                unitary_loss = torch.tensor(weight).to(self.device) * \
                               loss(computed_flows, ground_truth=gt_flows)
                train_loss = train_loss + unitary_loss

                # normalize loss by the number of videos in the test dataset and the bunch of epochs
                loss2print[i] += unitary_loss.item() / len(self.train_loader)

                i += 1

            #Bak-Propagation
            train_loss.backward()
            self.encDec_optimizer.step()

        return loss2print

    def train(self):
        """
        Main training loop
        :return:
        """
        #Train first encoder Decoder, once this has been converged, train update

        self.train_encDec()
        #self.train_update()

    def train_encDec(self):
        # Visualization purposes
        train_loss_names = ['Encoder-Decoder TRAIN loss---> ' +
                            l.__class__.__name__ for l in self.encDec_losses_fn] +\
                           ['Encoder-Decoder TRAIN loss---> TOTAL']
        val_loss_names = ['Encoder-Decoder VAL loss---> ' +
                           l.__class__.__name__ for l in self.encDec_losses_fn] +\
                        ['Encoder-Decoder VAL loss---> TOTAL']


        for epoch in tqdm(range(self.encDec_epoch+1, self.encDec_n_epochs),
                          initial = self.encDec_epoch, total=self.encDec_n_epochs,
                          desc="Epoch"):

            training_losses = self.train_one_epoch_encDec()
            # Add the total
            training_losses = training_losses + [sum(training_losses)]
            self.encDec_epoch = epoch

            validation_losses = self.eval_encDec(self.val_loader, verbose=True)
            validation_losses = validation_losses + [sum(validation_losses)]

            #Tensor board training
            for metric, title in zip(training_losses, train_loss_names):
                self.TB_writer.add_scalar(title, metric, epoch)

            # Tensor board validation
            for metric, title in zip(validation_losses, val_loss_names):
                self.TB_writer.add_scalar(title, metric, epoch)

            is_best = True
            if self.encDec_loss is not None:
                is_best = validation_losses[-1] < self.encDec_loss

            if is_best:
                self.encDec_loss = validation_losses[-1]

            #self.save_encDec_checkpoint(is_best=is_best)



    def eval_encDec(self, loader, verbose = False):
        # train mode
        self.encoder_decoder.eval()

        loss2print = [0] * len(self.encDec_losses_fn)

        nSec = 1
        for data in tqdm(loader, leave=False, desc='    Videos: '):
            _, flows, masks, _, gt_flows = data

            B, N, C, H, W = flows.shape

            # Remove the batch dimension (for pierrick architecture is needed B to be 1)
            flows = flows.view(B * N, C, H, W)
            gt_flows = gt_flows.view(B * N, C, H, W)

            self.encDec_optimizer.zero_grad()
            # Forward Pass
            computed_flows = self.encoder_decoder(flows)

            # Compute loss
            i = 0
            for loss, weight in zip(self.encDec_losses_fn, self.encDec_losses_weight):
                unitary_loss = torch.tensor(weight).to(self.device) * \
                               loss(computed_flows, ground_truth=gt_flows)

                # normalize loss by the number of videos in the test dataset and the bunch of epochs
                loss2print[i] += unitary_loss.item() / len(self.train_loader)

                i += 1

            verbose_images(self.val_out_images, prefix = 'encDec_sec_{}_'.format(str(nSec)),
                        input_flow=flows, computed_flow=computed_flows,
                        gt_flow=gt_flows)
            nSec += 1
        return loss2print



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