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
from ingestion import *

from utils.data_io import create_dir, verbose_images
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import shutil


class EncDec_update_agent_003(BaseAgent):
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
        self.update_loss = None


        self.mode = config.general.mode

        ## Member variables
        self.encDec_losses_fn = []
        self.encDec_losses_weight = []

        self.update_losses_fn = []
        self.update_losses_weight = []

        ######################
        # Folders
        self.exp_name = config.general.exp_name
        self.tb_stats_dir = join(config.verbose.tensorboard_root_dir, self.exp_name)
        self.checkpoint_dir = join(config.checkpoint.root_dir, self.exp_name)
        self.verbose_out_images = join(config.verbose.verbose_out_images, self.exp_name)

        create_dir(self.checkpoint_dir)
        create_dir(self.tb_stats_dir)
        create_dir(self.verbose_out_images)
        self.encDec_checkpoint_filename = join(self.checkpoint_dir, config.checkpoint.enc_dec_checkpoint_filename)
        self.update_checkpoint_filename = join(self.checkpoint_dir, config.checkpoint.update_checkpoint_filename)

        # Datasets
        self.set_data(config.data)

        # Models
        self.set_model(config.model)

        # Optimizers
        self.set_optimizer(config.optimizer)

        # Losses
        self.set_losses(config.losses)

        # Tensor Board writers
        self.TB_writer = SummaryWriter(join(config.verbose.tensorboard_root_dir, self.exp_name))

        #################################
        # If exists checkpoints, load them
        self.load_checkpoint(config.checkpoint)


    ########################################################################
    # SET FUNCTIONS
    def set_data(self, data_config):

        restaurant = globals()[data_config.restaurant]
        # TODO: pasar los parametros encapsulados en la estructura del json para poder usar diferences restaurantes sin tener
        # que cambiar el agente. Tener dos estructuras, una para el entreno y otra para el val
        train_data = restaurant(data_config.train_root_dir,
                                data_config.generic_mask_sequences_dir,
                                      GT=True,
                                      number_of_frames = data_config.number_of_frames
                                      )

        self.train_loader = DataLoader(train_data, batch_size=1, shuffle=True, drop_last=False)

        val_data = restaurant(data_config.val_root_dir,
                              data_config.generic_mask_sequences_dir,
                                    number_of_frames=data_config.number_of_frames,
                                    GT=True,
)
        self.val_loader = DataLoader(val_data, batch_size=1, shuffle=False, drop_last=False)

    def set_model(self, model_config):
        self.encoder_decoder = globals()[model_config.encDec]
        self.encoder_decoder = self.encoder_decoder().to(self.device)

        self.update = globals()[model_config.update]
        self.update = self.update(device=self.device).to(self.device)



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



        for loss, weight in zip(loss_config.encDec_losses.losses, loss_config.encDec_losses.weights):
            self.encDec_losses_fn.append(globals()[loss](device= self.device))
            self.encDec_losses_weight.append(weight)

        for loss, weight in zip(loss_config.update_losses.losses, loss_config.update_losses.weights):
            self.update_losses_fn.append(globals()[loss](device = self.device))
            self.update_losses_weight.append(weight)

    # END of SET FUNCTIONS
    #######################################################################


    def load_checkpoint(self, chk_config):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """

        if os.path.exists(self.encDec_checkpoint_filename):
            checkpoint = torch.load(self.encDec_checkpoint_filename, map_location='cpu')

            self.encoder_decoder.load_state_dict(checkpoint['model_state_dict'])
            self.encDec_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.encDec_epoch = checkpoint['epoch']
            self.encDec_loss = checkpoint['total_loss']

            print('** Checkpoint ' + self.encDec_checkpoint_filename + ' loaded \n')


        if os.path.exists(self.update_checkpoint_filename):
            checkpoint = torch.load(self.update_checkpoint_filename, map_location='cpu')
            self.update.load_state_dict(checkpoint['model_state_dict'])
            self.update_optimizer.load_state_dict((checkpoint['optimizer_state_dict']))
            self.update_epoch = checkpoint['epoch']
            self.update_loss = checkpoint['total_loss']

            print('** Checkpoint ' + self.update_checkpoint_filename + ' loaded \n')

    def save_checkpoint_encDec(self, is_best=False):
        chk = {
            'epoch': self.encDec_epoch,
            'model_state_dict': self.encoder_decoder.state_dict(),
            'optimizer_state_dict': self.encDec_optimizer.state_dict(),
            'device': self.device,
            'total_loss': self.encDec_loss
        }

        torch.save(chk, self.encDec_checkpoint_filename)

        if is_best:
            shutil.copyfile(self.encDec_checkpoint_filename,
                            self.encDec_checkpoint_filename[:-4] +'_best.tar')

        print('\n Encoder-Decoder Checkpoints saved')

    def save_checkpoint_update(self, is_best=False):
        chk = {
            'epoch': self.update_epoch,
            'model_state_dict': self.update.state_dict(),
            'optimizer_state_dict': self.update_optimizer.state_dict(),
            'device': self.device,
            'total_loss': self.update_loss
        }

        torch.save(chk, self.update_checkpoint_filename)

        if is_best:
            shutil.copyfile(self.update_checkpoint_filename,
                            self.update_checkpoint_filename[:-4] +'_best.tar')

        print('\n Update Checkpoints saved')
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

    def run_one_epoch_encDec(self, data_loader, training = True, verbose = False):
        # train mode
        if training:
            self.encoder_decoder.train()
        else:
            self.encoder_decoder.eval()

        loss2print = [0] * len(self.encDec_losses_fn)

        nSec = 1
        for data in tqdm(data_loader, leave=False, desc='    Videos: '):
            _, flows, masks, _, gt_flows = data

            B, N, C, H, W = flows.shape

            # Remove the batch dimension (for pierrick architecture is needed B to be 1)
            flows = flows.view(B * N, C, H, W)
            gt_flows = gt_flows.view(B * N, C, H, W)

            # place data on device
            flows = flows.to(self.device)
            gt_flows = gt_flows.to(self.device)

            if training:
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
                loss2print[i] += unitary_loss.item() / len(data_loader)

                i += 1

            if training:
                #Bak-Propagation
                train_loss.backward()
                self.encDec_optimizer.step()

            if verbose:
                verbose_images(self.verbose_out_images, prefix='encDec_sec_{}_'.format(str(nSec)),
                               input_flow=flows, computed_flow=computed_flows,
                               gt_flow=gt_flows)
                nSec += 1

        return loss2print

    def run_one_epoch_update(self,  data_loader, training = True, verbose = False):
        torch.autograd.set_detect_anomaly(True)

        if training:
            self.update.train()

        loss2print = [0] * len(self.update_losses_fn)

        nSec = 1
        for data in tqdm(data_loader, leave=False, desc='    Videos: '):
            _, iflows, masks, _, gt_flows = data

            B, N, C, H, W = iflows.shape

            # Remove the batch dimension (for pierrick architecture is needed B to be 1)
            iflows = iflows.view(B * N, C, H, W)
            # masks: 1 inside the hole
            masks = masks.view(B * N, 1, H, W)

            gt_flows = gt_flows.view(B * N, C, H, W)

            # place data on device
            iflows = iflows.to(self.device)
            masks = masks.to(self.device)
            gt_flows = gt_flows.to(self.device)

            # Initial confidence: 1 outside the mask (the hole), 0 inside
            initial_confidence = 1 - 1. * masks
            confidence = initial_confidence.clone()
            gained_confidence = initial_confidence

            computed_flows = iflows.clone()

            F = self.encoder_decoder.encode(iflows)

            step = -1



            #print("Numero Steps: ", step)
            loss2print = [0] * len(self.update_losses_fn)


            if training:
                self.update_optimizer.zero_grad()


            new_F = self.update((F, iflows, confidence))

            computed_flows = self.encoder_decoder.decode(new_F)

            train_loss = torch.tensor(0).to(self.device)
            i = 0
            for loss, weight in zip(self.update_losses_fn, self.update_losses_weight):
                unitary_loss = torch.tensor(weight).to(self.device) * \
                               loss(computed_flows, ground_truth=gt_flows)
                train_loss = train_loss + unitary_loss

                # normalize loss by the number of videos in the test dataset and the bunch of epochs
                loss2print[i] += unitary_loss.item() / (len(data_loader) )

                i += 1

            if training:
                #Back-Propagation
                train_loss.backward()
                self.update_optimizer.step()


            if verbose:
                verbose_images(self.verbose_out_images, prefix='update_sec_{}_'.format(str(nSec)),
                               input_flow=iflows, computed_flow=computed_flows,
                               gt_flow=gt_flows)
            nSec += 1

        return loss2print

    def train(self):
        """
        Main training loop
        :return:
        """
        #Train first encoder Decoder, once this has been converged, train update

        self.train_encDec()

        # freeze encoder/decoder
        for learn_param in self.encoder_decoder.parameters():
            learn_param.requires_grad = False

        for learn_param in self.encoder_decoder.parameters():
            learn_param.requires_grad = False

        self.encoder_decoder.eval()

        self.train_update()

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
                          desc="Enc/Dec Epoch"):

            training_losses = self.run_one_epoch_encDec(data_loader= self.train_loader, training = True, verbose = False)
            # Add the total
            training_losses = training_losses + [sum(training_losses)]
            self.encDec_epoch = epoch

            with torch.no_grad():
                validation_losses = self.run_one_epoch_encDec(data_loader = self.val_loader, training = False, verbose=True)

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

            self.save_checkpoint_encDec(is_best=is_best)


    def train_update(self):
        # Visualization purposes
        train_loss_names = ['Update TRAIN loss---> ' +
                            l.__class__.__name__ for l in self.update_losses_fn] + \
                           ['Update TRAIN loss---> TOTAL']
        val_loss_names = ['Update VAL loss---> ' +
                          l.__class__.__name__ for l in self.update_losses_fn] + \
                         ['Update VAL loss---> TOTAL']

        for epoch in tqdm(range(self.update_epoch + 1, self.update_n_epochs),
                          initial=self.update_epoch, total=self.update_n_epochs,
                          desc="Update Epoch"):

            training_losses = self.run_one_epoch_update(data_loader=self.train_loader, training=True, verbose=False)
            # Add the total
            training_losses = training_losses + [sum(training_losses)]
            self.update_epoch = epoch

            with torch.no_grad():
                validation_losses = self.run_one_epoch_update(data_loader=self.val_loader, training=False, verbose=True)

            validation_losses = validation_losses + [sum(validation_losses)]

            # Tensor board training
            for metric, title in zip(training_losses, train_loss_names):
                self.TB_writer.add_scalar(title, metric, epoch)

            # Tensor board validation
            for metric, title in zip(validation_losses, val_loss_names):
                self.TB_writer.add_scalar(title, metric, epoch)

            is_best = True
            if self.update_loss is not None:
                is_best = validation_losses[-1] < self.update_loss

            if is_best:
                self.update_loss = validation_losses[-1]

            self.save_checkpoint_update(is_best=is_best)


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