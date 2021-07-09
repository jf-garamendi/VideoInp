import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

from agents.encDec_update_001 import EncDec_update_agent_001
import torch
from tqdm import tqdm
import torchvision.transforms as T
from torch.utils.data import DataLoader
from ingestion import *
from utils.data_io import  verbose_images
class Ag005_EncDec_MultiScaleUpdate(EncDec_update_agent_001):
    def __init__(self, config):
        super().__init__(config)

    def set_data(self, data_config):
        restaurant = globals()[data_config.restaurant]
        # TODO: pasar los parametros encapsulados en la estructura del json para poder usar diferences restaurantes sin tener
        # que cambiar el agente. Tener dos estructuras, una para el entreno y otra para el val
        train_data = restaurant(data_config.train_root_dir,
                                data_config.train_generic_mask_sequences_dir,
                                GT=True,
                                number_of_frames=data_config.number_of_frames, nLevels= data_config.n_levels
                                )

        self.train_loader = DataLoader(train_data, batch_size=1, shuffle=True, drop_last=False)

        val_data = restaurant(data_config.val_root_dir,
                              data_config.val_generic_mask_sequences_dir,
                              number_of_frames=data_config.number_of_frames,
                              GT=True, nLevels= data_config.n_levels
                              )
        self.val_loader = DataLoader(val_data, batch_size=1, shuffle=False, drop_last=False)

    def run_one_epoch_update(self, data_loader, training=True, verbose=False):
        torch.autograd.set_detect_anomaly(True)

        if training:
            self.update.train()

        loss2print = [0] * len(self.update_losses_fn)

        nSec = 1
        for pyramid_data in tqdm(data_loader, leave=False, desc='    Videos: '):
            previous_scale_confidence = 0
            previous_scale_flow = 0
            #DEBUG
            pyramid_data = [ pyramid_data[i][-2:] for i in range(len(pyramid_data))]
            # end DEBUG
            for iflows, masks, gt_flows in zip(pyramid_data[1], pyramid_data[2], pyramid_data[4]):
                # from coarsest to finest
                B, N, C, H, W = iflows.shape

                # Remove the batch dimension (for pierrick architecture is needed B to be 1)

                # masks: 1 inside the hole
                masks = masks.view(B * N, 1, H, W)

                gt_flows = gt_flows.view(B * N, C, H, W)

                # place data on device
                iflows = iflows.to(self.device)
                masks = masks.to(self.device)
                gt_flows = gt_flows.to(self.device)

                # Initial confidence: 1 outside the mask (the hole), 0 inside
                initial_confidence = (1 - 1. * masks) + previous_scale_confidence*0.25
                initial_confidence = torch.clip(initial_confidence, 0, 1)

                iflows = iflows.view(B * N, C, H, W) + previous_scale_flow
                confidence = initial_confidence.clone()
                gained_confidence = initial_confidence

                computed_flows = iflows.clone()

                F = self.encoder_decoder.encode(iflows)

                step = -1

                max_num = 0
                while (step <= self.max_num_steps_update):
                    # print("Numero Steps: ", step)
                    loss2print = [0] * len(self.update_losses_fn)
                    step += 1
                    # print(step)

                    if training:
                        self.update_optimizer.zero_grad()

                    current_flow = computed_flows.clone().detach()
                    current_F = F.clone().detach()

                    new_F, confidence_new = self.update((current_F, current_flow, confidence))

                    gained_confidence = ((confidence_new > confidence) * confidence_new)

                    if gained_confidence.sum() > 0:
                        F = current_F * (confidence_new <= confidence) + new_F * (confidence_new > confidence)
                        computed_flows = self.encoder_decoder.decode(F)

                        train_loss = torch.tensor(0).to(self.device)
                        i = 0
                        for loss, weight in zip(self.update_losses_fn, self.update_losses_weight):
                            unitary_loss = torch.tensor(weight).to(self.device) * \
                                           loss(computed_flows, mask=gained_confidence, ground_truth=gt_flows)
                            train_loss = train_loss + unitary_loss

                            # normalize loss by the number of videos in the test dataset and the bunch of epochs
                            loss2print[i] += unitary_loss.item() / (len(data_loader))

                            i += 1

                        if training:
                            # Back-Propagation
                            train_loss.backward()
                            self.update_optimizer.step()

                        # mask update before next step
                        confidence = confidence_new.clone().detach()

                previous_scale_confidence = T.Resize(size=(H*2, W*2))(confidence) * 0.5
                previous_scale_flow = T.Resize(size=(H*2, W*2))( masks*computed_flows)


            if verbose:
                verbose_images(self.verbose_out_images, prefix='update_sec_{}_'.format(str(nSec)),
                               input_flow=iflows, computed_flow=computed_flows,
                               gt_flow=gt_flows)
            nSec += 1

        return loss2print
