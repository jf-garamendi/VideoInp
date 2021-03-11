import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

from training.VideoInp_DataSet import VideoInp_DataSet
from torch.utils.data import DataLoader

from model.iterative import Flow2features, Features2flow, Res_Update
from torchsummary import summary

from torch import optim, nn

from torchvision import transforms
import torch
from PIL import Image
from utils.data_io import create_dir
# TODO: Argumentos a linea de comandos
root_dir = '../data_t'

from tqdm import tqdm
import numpy as np

from utils.data_io import tensor_save_flow_and_img
from os.path import join

#Arguments


#device = torch.device("cuda:0")
DEVICE = torch.device("cpu")

################################################################

def train_encoder_decoder(encoder, decoder, train_loader, optim, loss_computer, n_epochs):
    # initialize loss
    loss_total = 0.0

    for epoch in range(n_epochs):
        for i, data in enumerate(train_loader):
            # get the input, data is a tuple composed by
            flows, mask, gt_flows = data
            B, N, C, H, W = flows.shape

            flows = flows.view(B * N, C, H, W)
            mask = mask.view(B * N, 1, H, W)
            gt_flows = gt_flows.view(B * N, C, H, W)

            #place data on device
            #flows.to(DEVICE)
            #mask.to(DEVICE)
            #gt_flows.to(DEVICE)

            # zero the parameter gradients
            optim.zero_grad()


            # Forward Pass
            computed_flows = decoder(encoder(flows))

            loss = loss_computer(computed_flows, gt_flows, torch.squeeze(mask))

            # Backward Pass
            loss.backward()
            optim.step()

            #update loss
            loss_total += loss.item()

            #print loss tatistics
            if (i) % 2 ==0:
                print('[Epoch number: %d, Mini-batchees: %5d] loss: %.3f' %(epoch +1, i+1, loss_total))
                loss_total = 0.0

                # save flow images
                folder = join('salida_entreno_enc_dec', 'computed_forward_flow')
                tensor_save_flow_and_img(computed_flows[:, 0:2, :, :], folder)

                folder = join('salida_entreno_enc_dec', 'GT_forward_flow')
                tensor_save_flow_and_img(gt_flows[:, 0:2, :, :], folder)

                folder = join('salida_entreno_enc_dec', 'input_forward_flow')
                tensor_save_flow_and_img(flows[:, 0:2, :, :], folder)


def train_all(flow2F, F2flow, update_net, train_loader, optimizer, loss_computer, n_epochs):
    for epoch in range(n_epochs):
        print('starting epoch {:d}'.format(epoch))

        it = 0
        for i, (flows_i, mask_i, gt_flows_i) in enumerate(tqdm(train_loader)):
            it += 1

            initial_confidence = 1 - mask_i
            current_confidence = initial_confidence

            B, N, C, H, W = flows_i.shape

            xx, yy = torch.meshgrid(torch.arange(H), torch.arange(W))
            ind = torch.stack((yy, xx), dim=-1)
            ind = ind.repeat(B, 1, 1, 1)

            #GT_F = flow2F(gt_flows_i.view(B * N, C, H, W)).view(B, N, 32, H, W)

            decoded_flows = flows_i
            #F = flow2F(flows_i.view(B * N, C, H, W)).view(B, N, 32, H, W)
            for step in tqdm(range(20), desc='## Step  ##', position=0):
                optimizer.zero_grad()
                #decoded_flows = F2flow(F.view(B * N, -1, H, W))
                F = flow2F(decoded_flows.view(B * N, C, H, W)).view(B, N, 32, H, W)
                #F = flow2F(flows_i.view(B * N, C, H, W)).view(B, N, 32, H, W)
                all_frames_flow_from_features = F2flow(F.view(B * N, 32, H, W)).view(B, N, C, H, W)

                confidence_new = current_confidence * 0.

                new_F = F * 0
                #for n_frame in tqdm(range(N), desc='   Frame', position=1, leave=False):
                for n_frame in range(N):
                    flow_from_features = all_frames_flow_from_features[:, n_frame]

                    ## warping
                    if n_frame + 1 < N:
                        grid_f = flow_from_features[:, :2, :, :].permute((0, 2, 3, 1)) + ind

                        # Normalize the coordinates to the square [-1,1]
                        grid_f = (2 * grid_f / torch.tensor([W, H]).cpu().view(1, 1, 1, 2)) - 1

                        # warp ## FORWARD ##
                        F_f = torch.nn.functional.grid_sample(F[:, n_frame + 1, :, :, :], grid_f,
                                                              mode='bilinear', padding_mode='border',
                                                              align_corners=False)
                        confidence_f = torch.clamp(
                            torch.nn.functional.grid_sample(current_confidence[:, n_frame + 1, :, :, :], grid_f,
                                                            mode='bilinear',
                                                            padding_mode='border', align_corners=False), 0, 1)
                    else:
                        F_f = 0. * F[:, n_frame]
                        confidence_f = 0. * current_confidence[:, n_frame]

                    if n_frame - 1 >= 0:

                        grid_b = flow_from_features[:, 2:].permute(
                            (0, 2, 3, 1)) + ind  # compute backward flow from features

                        # Normalize the coordinates to the square [-1,1]
                        grid_b = (2 * grid_b / torch.tensor([W, H]).cpu().view(1, 1, 1, 2)) - 1

                        # warp  ## BACKWARD ##
                        F_b = torch.nn.functional.grid_sample(F[:, n_frame - 1, :, :, :], grid_b, mode='bilinear',
                                                              padding_mode='border',
                                                              align_corners=False)
                        confidence_b = torch.clamp(
                            torch.nn.functional.grid_sample(current_confidence[:, n_frame - 1, :, :, :], grid_b,
                                                            mode='bilinear', padding_mode='border',
                                                            align_corners=False), 0,1)
                    else:
                        F_b = 0. * F[:, n_frame]
                        confidence_b = 0. * current_confidence[:, n_frame]
                    # End warping

                    # input of the update network is the concatenation of the obtained features from this frame and the neighboring ones
                    x = torch.cat((F_b, F[:, n_frame], F_f), dim=1)
                    confidence_in = torch.cat(((confidence_b).repeat(1, F.shape[2], 1, 1),
                                               current_confidence[:, n_frame].repeat(1, F.shape[2], 1, 1),
                                               (confidence_f).repeat(1, F.shape[2], 1, 1)),
                                              dim=1)  # same goes for the input mask

                    ### UPDATE ###
                    new_F[:, n_frame], confidence_new[:, n_frame] = update_net(x, confidence_in)  # Update

                    # force the initially confident pixels to stay confident, because a decay can be observed
                    # depending on the update rule of the partial convolution
                    confidence_new[:, n_frame][initial_confidence[:, n_frame] == 1] = 1.

                    #Print Results
                    folder = join('salida_entreno_all', 'mask')
                    create_dir(folder)
                    m_np = confidence_new.view(B*N, H, W).numpy()
                    m_pil = Image.fromarray(255*np.squeeze(m_np[1,:,:]))
                    if m_pil.mode != 'RGB':
                        m_pil = m_pil.convert('RGB')
                    m_pil.save(folder + '/{:04d}.png'.format(n_frame))

                F = F * (confidence_new <= current_confidence) + new_F * (confidence_new > current_confidence)
                decoded_flows = F2flow(F.view(B * N, -1, H, W))


                pointwise_l1_error = torch.abs(gt_flows_i.view(B * N, -1, H, W) - decoded_flows).mean(dim=1)
                #pointwise_l1_error = torch.abs(GT_F.view(B * N, -1, H, W) - new_F).mean(dim=1)
                loss_1 = pointwise_l1_error[current_confidence.view(N * B, H, W) == 1].mean()


                decoded_flows = decoded_flows.view(B, N, C, H, W)

                # loss2 is the sum of terms from each step of the iterative scheme
                # we compute loss2 only on the pixels that gained confidence and we weight the result with the new confidence
                s = ((confidence_new > current_confidence) * confidence_new).sum()

                if s != 0:
                    loss_2 = (1 - np.exp(-it / (1000 * (step + 1)))) * (
                            pointwise_l1_error * ((confidence_new > current_confidence) * confidence_new)).sum() / s
                    # we add the weight  (1-np.exp(-it/(1000*(step+1)))) that makes loss2 be 0 for the first batch iteration, then around 1000, we account for the first step of the iterative scheme, and every 1000, another step is smoothly taken into account.
                    # justification: in the beginning, we want the translation between fows and features to converge before using the predicted flows to update the features. Alos, since the update is recurrent, we want to train well the first steps of it before adding a new one



                    # we handcraft the new feature volume
                else:
                    loss_2 = loss_1 * 0

                #F = F * (confidence_new <= current_confidence) + new_F * (confidence_new > current_confidence)

                current_confidence = confidence_new * 1.  # mask update before next step


                total_loss = (1. * loss_1 + 1. * loss_2)
                print('Loss: ', total_loss.item())
                total_loss.backward()  # weighting of the loss
                optimizer.step()
                decoded_flows = decoded_flows *1

            # print loss tatistics
            #print('[Epoch number: %d, Mini-batchees: %5d] loss: %.3f' % (epoch + 1, i + 1, total_loss))
            #loss_total = 0.0

            # save flow images
            folder = join('salida_entreno_all', 'computed_forward_flow')
            tensor_save_flow_and_img(decoded_flows.view(B * N, C, H, W)[:, 0:2, :, :], folder)

            folder = join('salida_entreno_all', 'GT_forward_flow')
            tensor_save_flow_and_img(gt_flows_i.view(B * N, C, H, W)[:, 0:2, :, :], folder)

            folder = join('salida_entreno_all', 'input_forward_flow')
            tensor_save_flow_and_img(flows_i.view(B * N, C, H, W)[:, 0:2, :, :], folder)





'''
def train_update(encoder, decoder, update, train_loader, optim, loss_computer, n_epochs):
    oss_total = 0.0

    for epoch in range(n_epochs):
        for i, data in enumerate(train_loader):
            # get the input, data is a tuple composed by
            flows, mask, gt_flows = data
            B, N, C, H, W = flows.shape

            flows = flows.view(B * N, C, H, W)
            mask = mask.view(B * N, 1, H, W)
            gt_flows = gt_flows.view(B * N, C, H, W)

            # place data on device
            # flows.to(DEVICE)
            # mask.to(DEVICE)
            # gt_flows.to(DEVICE)

            # zero the parameter gradients
            optim.zero_grad()

            hole = 1 - mask
            initial_confidence = 1 - mask
            current_confidence = initial_confidence

            xx, yy = torch.meshgrid(torch.arange(H), torch.arange(W))
            ind = torch.stack((yy, xx), dim=-1)
            ind = ind.repeat(B, 1, 1, 1)

            decoded_flows_old = flows

            for step in tqdm(range(40), desc='## Step ##', position=0):
                mask = mask_i.detach().clone()
                gt_flows = gt_flows_i.detach().clone()
                decoded_flows = decoded_flows_old

                optimizer.zero_grad()

                F = flow2F(decoded_flows)
                
                confidence_new = current_confidence * 0.

                new_F = F * 0
                for n_frame in tqdm(range(N), desc='   Frame', position=1, leave=False):
                    flow_from_features = decoded_flows[n_frame, :, :, :]

                    if n_frame + 1 < N:
                        grid_f = flow_from_features[:, :2, :, :].permute((0, 2, 3, 1)) + ind

                        # Normalize the coordinates to the square [-1,1]
                        grid_f = (2 * grid_f / torch.tensor([W, H]).cpu().view(1, 1, 1, 2)) - 1

                        # warp ## FORWARD ##
                        F_f = torch.nn.functional.grid_sample(F[n_frame + 1, :, :, :], grid_f,
                                                              mode='bilinear', padding_mode='border',
                                                              align_corners=False)
                        confidence_f = torch.clamp(
                            torch.nn.functional.grid_sample(current_confidence[n_frame + 1, :, :, :], grid_f,
                                                            mode='bilinear',
                                                            padding_mode='border', align_corners=False), 0, 1)
                    else:
                        F_f = 0. * F[n_frame]
                        confidence_f = 0. * current_confidence[n_frame]

                    if n_frame - 1 >= 0:

                        grid_b = flow_from_features[:, 2:,:,:].permute(
                            (0, 2, 3, 1)) + ind  # compute backward flow from features

                        # Normalize the coordinates to the square [-1,1]
                        grid_b = (2 * grid_b / torch.tensor([W, H]).cpu().view(1, 1, 1, 2)) - 1

                        # warp  ## BACKWARD ##
                        F_b = torch.nn.functional.grid_sample(F[n_frame - 1, :, :, :], grid_b, mode='bilinear',
                                                              padding_mode='border',
                                                              align_corners=False)
                        confidence_b = torch.clamp(
                            torch.nn.functional.grid_sample(current_confidence[n_frame - 1, :, :, :], grid_b,
                                                            mode='bilinear', padding_mode='border',
                                                            align_corners=False), 0,
                            1)
                    else:
                        F_b = 0. * F[n_frame]
                        confidence_b = 0. * current_confidence[:, n_frame]
                    # --
                    # input of the update network is the concatenation of the obtained features from this frame and the neighboring ones
                    x = torch.cat((F_b, F[:, n_frame], F_f), dim=1)
                    confidence_in = torch.cat(((confidence_b).repeat( F.shape[2], 1, 1),
                                               current_confidence[:, n_frame].repeat(F.shape[2], 1, 1),
                                               (confidence_f).repeat( F.shape[2], 1, 1)),
                                              dim=1)  # same goes for the input mask

                    new_F[n_frame], confidence_new[n_frame] = update_net(x, confidence_in)  # Update

                    # force the initially confident pixels to stay confident, because a decay can be observed
                    # depending on the update rule of the partial convolution
                    confidence_new[ n_frame][hole[ n_frame] == 1] = 1.

                # loss2 is the sum of terms from each step of the iterative scheme
                decoded_flows = F2flow(new_F.view( N, -1, H, W))
                decoded_flows_old = decoded_flows.detach().clone()
                pointwise_l1_error = torch.abs(gt_flows - decoded_flows)

                # we compute loss2 only on the pixels that gained confidence and we weight the result with the new confidence
                s = ((confidence_new > current_confidence) * confidence_new).sum()
                if s != 0:
                    loss_2 =  ( pointwise_l1_error * ((confidence_new > current_confidence) * confidence_new)).sum() / s
                    # we add the weight  (1-np.exp(-it/(1000*(step+1)))) that makes loss2 be 0 for the first batch iteration, then around 1000, we account for the first step of the iterative scheme, and every 1000, another step is smoothly taken into account.
                    # justification: in the beginning, we want the translation between fows and features to converge before using the predicted flows to update the features. Alos, since the update is recurrent, we want to train well the first steps of it before adding a new one

                pointwise_l1_error = pointwise_l1_error.view(B * N, C, H, W).mean(dim=1)
                loss_1 = pointwise_l1_error[current_confidence.view(N * B, H, W) == 1].mean()
                # we handcraft the new feature volume
                F = F * (confidence_new <= current_confidence) + new_F * (confidence_new > current_confidence)

                current_confidence = confidence_new * 1.  # mask update before next step

                total_loss = (1. * loss_1 + 1. * loss_2)
                print('Loss: ', total_loss.item())
                total_loss.backward()  # weighting of the loss
                optimizer.step()
'''

def pierrcik_L1_loss(flow, gt_flow, mask):
    pointwise_l1_error = (torch.abs(gt_flow - flow) ** 1).mean(dim=1)
    loss_1 = pointwise_l1_error[mask == 1].mean()

    return loss_1



if __name__ == '__main__':


    train_data = VideoInp_DataSet(root_dir)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, drop_last=True)

    # Net Models
    flow2F = Flow2features()
    F2flow = Features2flow()
    update_net = Res_Update()

    flow2F.to(DEVICE)
    F2flow.to(DEVICE)
    update_net.to(DEVICE)

    flow2F.train()
    F2flow.train()
    update_net.train()

    # Optimazer
    optimizer = optim.Adam( list(F2flow.parameters()) + list(flow2F.parameters()),
                           lr=1e-4,
                           betas=(0.9, 0.999),
                           weight_decay=0.00004)

    #loss
    loss_computer = pierrcik_L1_loss
    train_encoder_decoder(flow2F, F2flow, train_loader, optimizer, loss_computer, 3)

    for param in flow2F.parameters():
        param.requires_grad = False

    for param in F2flow.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(update_net.parameters(),
                           lr=1e-4,
                           betas=(0.9, 0.999),
                           weight_decay=0.00004)
    torch.autograd.set_detect_anomaly(True)
    train_all(flow2F, F2flow, update_net, train_loader, optimizer, loss_computer, 3)


#-----------------------------------------
'''
for epoch in range(1000):
    print('starting epoch {:d}'.format(epoch))

    flow2F.cpu().train()
    F2flow.cpu().train()
    update_net.cpu().train()

    for i, (flows_i, mask_i, gt_flows_i) in enumerate(tqdm(train_loader)):

        it += 1
        hole = 1 - mask_i
        initial_confidence = 1 -  mask_i
        current_confidence =  initial_confidence

        B, N, C, H, W = flows_i.shape

        xx, yy = torch.meshgrid(torch.arange(H), torch.arange(W))
        ind = torch.stack((yy, xx), dim=-1)
        ind = ind.repeat(B, 1, 1, 1)


        decoded_flows_old = flows_i.detach().clone()

        for step in tqdm(range(40), desc='## Step ##', position=0):
            mask = mask_i.detach().clone()
            gt_flows= gt_flows_i.detach().clone()
            decoded_flows = decoded_flows_old

            optimizer.zero_grad()

            F = flow2F(decoded_flows.view(B * N, C, H, W)).view(B, N, 32, H, W)
            #reconstruction_flows = F2flow(F.view(B * N, -1, H, W))

            # pointwise_l1_error = (torch.abs(gt_flows.view(B * N, C, H, W) - reconstruction_flows) ** 1).mean(dim=1)
            # loss_1 = pointwise_l1_error[mask.view(N * B, H, W) == 1].mean()
            # loss_2 = 0.

            confidence_new = current_confidence * 0.

            new_F=F*0
            for n_frame in tqdm(range(N), desc='   Frame', position=1, leave=False):
                flow_from_features = decoded_flows[:, n_frame, :, :, :]


                if n_frame +1 < N:
                    grid_f = flow_from_features[:, :2, :, :].permute((0, 2, 3, 1)) + ind

                    # Normalize the coordinates to the square [-1,1]
                    grid_f = (2 * grid_f / torch.tensor([W, H]).cpu().view(1, 1, 1, 2)) - 1

                    # warp ## FORWARD ##
                    F_f = torch.nn.functional.grid_sample(F[:, n_frame + 1, :, :, :], grid_f,
                                                          mode='bilinear', padding_mode='border', align_corners=False)
                    confidence_f = torch.clamp(
                        torch.nn.functional.grid_sample(current_confidence[:, n_frame + 1, :, :, :], grid_f,
                                                        mode='bilinear',
                                                        padding_mode='border', align_corners=False), 0, 1)
                else:
                    F_f = 0. * F[:, n_frame]
                    confidence_f = 0. * current_confidence[:, n_frame]

                if n_frame - 1 >= 0:

                    grid_b = flow_from_features[:, 2:].permute(
                        (0, 2, 3, 1)) + ind  # compute backward flow from features

                    # Normalize the coordinates to the square [-1,1]
                    grid_b = (2 * grid_b / torch.tensor([W, H]).cpu().view(1, 1, 1, 2)) - 1

                    # warp  ## BACKWARD ##
                    F_b = torch.nn.functional.grid_sample(F[:, n_frame - 1, :, :, :], grid_b, mode='bilinear',
                                                          padding_mode='border',
                                                          align_corners=False)
                    confidence_b = torch.clamp(
                        torch.nn.functional.grid_sample(current_confidence[:, n_frame - 1, :, :, :], grid_b,
                                                        mode='bilinear', padding_mode='border', align_corners=False), 0,
                        1)
                else:
                    F_b = 0. * F[:, n_frame]
                    confidence_b = 0. * current_confidence[:, n_frame]
                #--
                # input of the update network is the concatenation of the obtained features from this frame and the neighboring ones
                x = torch.cat((F_b, F[:, n_frame], F_f), dim=1)
                confidence_in = torch.cat(((confidence_b).repeat(1, F.shape[2] , 1, 1),
                                           current_confidence[:, n_frame].repeat(1, F.shape[2] , 1, 1),
                                           (confidence_f).repeat(1, F.shape[2] , 1, 1)),
                                          dim=1)  # same goes for the input mask

                new_F[:, n_frame], confidence_new[:, n_frame] = update_net(x, confidence_in)  # Update

                # force the initially confident pixels to stay confident, because a decay can be observed
                # depending on the update rule of the partial convolution
                confidence_new[:, n_frame][hole[:, n_frame] == 1] = 1.

            # loss2 is the sum of terms from each step of the iterative scheme
            decoded_flows = F2flow(new_F.view(B * N, -1, H, W)).view(B , N,  C, H, W)
            decoded_flows_old = decoded_flows.detach().clone()
            pointwise_l1_error = torch.abs(gt_flows - decoded_flows)

            # we compute loss2 only on the pixels that gained confidence and we weight the result with the new confidence
            s = ((confidence_new > current_confidence) * confidence_new).sum()
            if s != 0:
                loss_2 = (1 - np.exp(-it / (1000 * (step + 1)))) * (
                            pointwise_l1_error * ((confidence_new > current_confidence) * confidence_new)).sum() / s
                # we add the weight  (1-np.exp(-it/(1000*(step+1)))) that makes loss2 be 0 for the first batch iteration, then around 1000, we account for the first step of the iterative scheme, and every 1000, another step is smoothly taken into account.
                # justification: in the beginning, we want the translation between fows and features to converge before using the predicted flows to update the features. Alos, since the update is recurrent, we want to train well the first steps of it before adding a new one

            pointwise_l1_error = pointwise_l1_error.view(B*N, C, H, W).mean(dim=1)
            loss_1 = pointwise_l1_error[current_confidence.view(N*B,H,W)==1].mean()
            # we handcraft the new feature volume
            F = F * (confidence_new <= current_confidence) + new_F * (confidence_new > current_confidence)

            current_confidence = confidence_new * 1.  # mask update before next step

            total_loss = (1. * loss_1 + 1. * loss_2)
            print('Loss: ', total_loss.item())
            total_loss.backward()  # weighting of the loss
            optimizer.step()


            #loss['flow_l1_reconstruction'] = loss_1.detach()
            #loss['masked_l1_prediction_error'] = loss_2.detach()
'''