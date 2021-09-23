import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

from agents.encDec_update_001 import EncDec_update_agent_001
import torch
from tqdm import tqdm
import torchvision.transforms as T

from ingestion import *
from utils.data_io import verbose_images, tensor_save_flow_and_img, save_mask_tensor_as_img
import numpy as np

class Ag005_EncDec_MultiScaleUpdate(EncDec_update_agent_001):
    def __init__(self, config):
        super().__init__(config)

        np.random.seed(2021)
        torch.manual_seed(2021)


    def run_one_epoch_update(self, data_loader, training=True, verbose=False):
        torch.autograd.set_detect_anomaly(True)

        if training:
            self.update.train()

        loss2print = [0] * len(self.update_losses_fn)

        nSec = 1
        for sec_name, pyramid_data in tqdm(data_loader, leave=False, desc='    Videos: '):
            previous_scale_confidence = 0
            previous_scale_flow = 0
            #DEBUG
            # choose (harcoded) the number of levels in the pyramid_data
            pyramid_data = [ pyramid_data[i][-2:] for i in range(len(pyramid_data))]
            # end DEBUG
            for iflows_o, masks, gt_frames, gt_flows in zip(pyramid_data[1], pyramid_data[2], pyramid_data[3], pyramid_data[4]):
                # from coarsest to finest
                B, N, C, H, W = iflows_o.shape

                # Remove the batch dimension (for pierrick architecture is needed B to be 1)

                # masks: 1 inside the hole
                masks = masks.view(B * N, 1, H, W)
                gt_frames = gt_frames.view(B * N, 3, H, W)

                gt_flows = gt_flows.view(B * N, C, H, W)
                iflows_o = iflows_o.view(B * N, C, H, W)

                # place data on device
                iflows_o = iflows_o.to(self.device)
                masks = masks.to(self.device)
                gt_flows = gt_flows.to(self.device)
                gt_frames = gt_frames.to(self.device)

                # Initial confidence: 1 outside the mask (the hole), 0 inside
                initial_confidence = (1 - 1. * masks) + previous_scale_confidence * 0.5
                initial_confidence = torch.clip(initial_confidence, 0, 1)

                uno = torch.Tensor([1]).to(self.device)
                iflows_a = (1-masks.type(torch.FloatTensor)).to(self.device)*iflows_o
                iflows_b = previous_scale_flow * (masks.type(torch.FloatTensor)).to(self.device)
                iflows = iflows_a + iflows_b

                '''
                tensor_save_flow_and_img(iflows_a[:, 0:2, :, :], "./borrar/input_flow_a")
                if previous_scale_flow is not 0:
                    tensor_save_flow_and_img(iflows_b[:, 0:2, :, :], "./borrar/input_flow_b")
                    save_mask_tensor_as_img(previous_scale_confidence, "./borrar/previous_scale_confidence")
                    tensor_save_flow_and_img(computed_flows[:, 0:2, :, :], "./borrar/grueso")
                    tensor_save_flow_and_img(previous_scale_flow[:, 0:2, :, :], "./borrar/fino")

                tensor_save_flow_and_img(iflows[:, 0:2, :, :], "./borrar/input_flow")
                tensor_save_flow_and_img(iflows_o[:, 0:2, :, :], "./borrar/input_flow_orig")
                save_mask_tensor_as_img(masks, "./borrar/masks")
                save_mask_tensor_as_img(initial_confidence, "./borrar/initial_confidence")
                '''


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

                previous_scale_confidence = T.Resize(size=(H*2, W*2))(confidence)
                previous_scale_flow = 2*T.Resize(size=(H*2, W*2))(computed_flows)



            if verbose:
                verbose_images(self.verbose_out_images, prefix='update_sec_$'+sec_name[0]+'$',
                               input_flow=iflows, computed_flow=computed_flows,
                               gt_flow=gt_flows, frames=gt_frames , masks=masks)
            nSec += 1

        return loss2print
