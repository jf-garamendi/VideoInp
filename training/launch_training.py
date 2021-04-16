import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

from training.VideoInp_DataSet import VideoInp_DataSet
from torch.utils.data import DataLoader

from model.iterative import Flow2features, Features2flow, Res_Update

from torch import optim, nn

from torchvision import transforms
import torch
import torchvision
from PIL import Image
from utils.data_io import create_dir


from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from utils.flow_viz import flow_to_image

from tqdm import tqdm
import numpy as np

from utils.data_io import tensor_save_flow_and_img
from os.path import join
from torchviz import make_dot

from utils.utils_from_FGVC.from_flow_to_frame import from_flow_to_frame_seamless, from_flow_to_frame

import training.training_parameters as param
# TODO: MUY IMPORTANTE--> Mover la lógica enc/dec/update a los ficheros del modelo ( model/iterative.py)

## Device where running
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("DEVICE: ", DEVICE)

################################################################
#TODO: Auxiliary functions that should go to a file
def flow_tensor_to_image_tensor_list(flow):
    # flow: nd_array of size (N,4,H,W)

    images = []
    for i in range(flow.shape[0]):
        flow_frame = torch.squeeze(flow[i, :, :, :]).permute((1, 2, 0))
        flow_frame = flow_frame.detach().cpu().numpy()

        flow_img = flow_to_image(flow_frame)

        images.append(torch.from_numpy(flow_img).permute(2,0,1))

    return images

def plot_optical_flow(flow, writer, caption=''):
    fwd_flow_images = flow_tensor_to_image_tensor_list(flow[:, :2])
    bwd_flow_images = flow_tensor_to_image_tensor_list(flow[:, 2:])
    img_grid = torchvision.utils.make_grid(fwd_flow_images+bwd_flow_images, nrow=len(fwd_flow_images))

    img_grid = img_grid.numpy()

    plt.imshow(np.transpose(img_grid, (1,2,0)))

    writer.add_image(caption, img_grid)

    return

def show_statistics(iter, verbose_dir, metrics_to_show, titles, caption, writer, input_flow_list=[],
                    computed_flow_list=[], gt_flow_list=[], computed_frames_list=[]):

    for metric, title in zip(metrics_to_show, titles):
        writer.add_scalar(title, metric, iter)
        print('\t\t' + title +'  : %.3f' % (metric))

    #plot_optical_flow(computed_flow, writer, caption + " : Computed Flows")
    #plot_optical_flow(gt_flow, writer, caption + " : Ground Truth Flows")


    # save Forward flow images
    i = 0
    for input_flow in input_flow_list:
        folder = join(verbose_dir, caption + 'input_forward_flow_sec#' + str(i) )
        tensor_save_flow_and_img(input_flow[:, 0:2, :, :], folder)

        # save Backward flow images
        folder = join(verbose_dir,  caption + 'input_backward_flow_sec#' + str(i))
        tensor_save_flow_and_img(input_flow[:, 2:, :, :], folder)

        i += 1

    i = 0
    for  computed_flow in computed_flow_list:
        folder = join(verbose_dir, caption + 'computed_forward_flow_sec#' + str(i))
        tensor_save_flow_and_img(computed_flow[:, 0:2, :, :], folder)

        folder = join(verbose_dir, caption + 'computed_backward_flow_sec#' + str(i))
        tensor_save_flow_and_img(computed_flow[:, 2:, :, :], folder)

        i +=1

    i = 0
    for gt_flow in gt_flow_list:
        folder = join(verbose_dir, caption + 'GT_forward_flow_sec#' + str(i))
        tensor_save_flow_and_img(gt_flow[:, 0:2, :, :], folder)

        folder = join(verbose_dir, caption + 'GT_backward_flow_sec#' + str(i))
        tensor_save_flow_and_img(gt_flow[:, 2:, :, :], folder)

        i +=1

    i=0
    for computed_frames in computed_frames_list:
        folder = join(verbose_dir, caption + 'warped_frames_sec#' + str(i))
        create_dir(folder)
        for n_frame in range(computed_frames.shape[3]):
            frame_blend = computed_frames[:, :, :, n_frame]
            m_pil = Image.fromarray((255 * np.squeeze(frame_blend)).astype(np.uint8))
            if m_pil.mode != 'RGB':
                m_pil = m_pil.convert('RGB')
            m_pil.save(folder + '/{:04d}_.png'.format(n_frame))


#############################
def train_encoder_decoder(encoder, decoder, train_loader, test_loader, optim, losses,  ini_epoch=0, final_epoch=100,
                          TB_writer = None, chk_path=None):

    #Visualization purposes
    train_loss_names = ['Encoder-Decoder train loss ' +
                        l.__name__ for l in losses['losses_list']]
    test_loss_names = ['Encoder-Decoder test loss ' +
                       l.__name__ for l in losses['losses_list']]
    train_loss2print = [0] * len(losses['losses_list'])

    for epoch in range(ini_epoch+1, final_epoch+1):
        for data in train_loader:
            # get the input, data is a tuple composed by
            _, flows, masks, _, gt_flows = data

            B, N, C, H, W = flows.shape

            # Remove the batch dimension (for pierrick architecture is needed B to be 1)
            flows = flows.view(B * N, C, H, W)
            gt_flows = gt_flows.view(B * N, C, H, W)

            # place data on device
            flows = flows.to(DEVICE)
            gt_flows = gt_flows.to(DEVICE)

            # zero the parameter gradients
            optim.zero_grad()

            # Forward Pass
            computed_flows = decoder(encoder(flows))

            train_loss = torch.tensor(0).to(DEVICE)
            i=0
            for loss, weight in zip(losses['losses_list'], losses['weights_list']):
                unitary_loss = torch.tensor(weight).to(DEVICE) * \
                               loss(computed_flows, ground_truth=gt_flows)
                train_loss = train_loss + unitary_loss

                ## update loss to print
                # normalize loss by the number of videos in the test dataset and the bunch of epochs
                train_loss2print[i] += unitary_loss.item() /  ( len(train_loader) * param.SHOWING_N_ITER )

                i += 1


            # Backward Pass
            train_loss.backward()
            optim.step()

        # print loss statistics
        if (epoch % param.SHOWING_N_ITER == 0) and (TB_writer is not None):
            print(' [Epoch for Encoder-Decoder %5d]' % epoch)

            # TODO: Moverlo a una funcion
            verbose_dir = join(param.VERBOSE_ROOT_DIR, param.TRAINING_NAME)

            ## Print Train losses
            scalars_to_show = train_loss2print + [sum(train_loss2print)]  # + concatenates
            name_of_scalars = train_loss_names + ['Encoder-Decoder Total Train Loss']
            show_statistics(epoch, verbose_dir, scalars_to_show, name_of_scalars, '', TB_writer)
            #Prepare for the next bunch of epochs
            train_loss2print = [0] * len(losses['losses_list'])

            # Print Test Losses
            # each component corresponds to a loss and it is the sum of the loses of all videos for the given loss
            test_loss = [0] * len(losses['losses_list'])

            #each component correspond to a video in the dataset
            flows_to_print = []
            computed_flows_to_print = []

            encoder.eval()
            decoder.eval()

            with torch.no_grad():
                for _, flows, _, _, gt_flows in test_loader:
                    B, N, C, H, W = flows.shape

                    flows = flows.view(B * N, C, H, W)
                    gt_flows = gt_flows.view(B * N, C, H, W)

                    # place data on device
                    flows = flows.to(DEVICE)
                    gt_flows = gt_flows.to(DEVICE)

                    computed_flows = decoder(encoder(flows))

                    flows_to_print.append(gt_flows)
                    computed_flows_to_print.append(computed_flows)

                    i = 0
                    for loss, weight in zip(losses['losses_list'], losses['weights_list']):
                        unitary_loss = weight * loss(computed_flows, ground_truth=gt_flows).item()

                        # normalize loss by the number of videos in the test dataset
                        test_loss[i] += unitary_loss / len(test_loader)

                        i+=1

            scalars_to_show =  test_loss + [sum(test_loss) ] # + concatenates
            name_of_scalars =  test_loss_names + ['Encoder-Decoder Total Test Loss']

            show_statistics(epoch, verbose_dir, scalars_to_show, name_of_scalars,  'encDec_', TB_writer,
                            input_flow_list=flows_to_print, computed_flow_list=computed_flows_to_print)

            encoder.train()
            decoder.train()


        if (epoch % param.SAVING_N_ITER == 0) and (chk_path is not None):
            # save checkpoint
            chk = {
                'epoch': epoch,
                'enc_state_dict': encoder.state_dict(),
                'dec_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'device': DEVICE
            }
            torch.save(chk, chk_path)


def train_update(flow2F, F2flow, update_net, train_loader, test_loader,
              optimizer,  losses,
              ini_epoch=0, final_epoch=100, TB_writer =None, chk_path=None):
    # Visualization purposes
    train_loss_names = ['Update train loss ' +
                        l.__name__ for l in losses['losses_list']]
    test_loss_names = ['Update test loss ' +
                       l.__name__ for l in losses['losses_list']]
    train_loss2print = [0] * len(losses['losses_list'])

    for epoch in range(ini_epoch+1, final_epoch+1):
        for _, iflows, masks, _, gt_flows in train_loader:
            # Remove the batch dimension (for pierrick architecture is needed B to be 1)
            B, N, C, H, W = iflows.shape

            # Remove the batch dimension (for pierrick architecture is needed B to be 1)
            iflows = iflows.view(B * N, C, H, W)
            # masks: 1 inside the hole
            masks = masks.view(B * N, 1, H, W)

            gt_flows = gt_flows.view(B * N, C, H, W)

            # place data on device
            iflows = iflows.to(DEVICE)
            masks = masks.to(DEVICE)
            gt_flows = gt_flows.to(DEVICE)

            # Initial confidence: 1 outside the mask (the hole), 0 inside
            initial_confidence = 1 - 1. * masks
            confidence = initial_confidence.clone()
            gained_confidence = initial_confidence

            new_flow = iflows.clone()

            F = flow2F(iflows)

            step = -1
            while (gained_confidence.sum() >0) and (step <= param.max_num_steps):
                step += 1
                #print(step)

                optimizer.zero_grad()
                current_flow = new_flow.clone().detach()
                current_F = F.clone().detach()

                new_F, confidence_new = update_step(update_net, current_flow, current_F, confidence)
                gained_confidence = ((confidence_new > confidence) * confidence_new)

                if gained_confidence.sum() > 0 :
                    F = current_F * (confidence_new <= confidence) + new_F * (confidence_new > confidence)
                    new_flow = F2flow(F)

                    train_total_loss = torch.tensor(0).to(DEVICE)
                    i=0
                    for loss, weight in zip(losses['losses_list'], losses['weights_list']):
                        unitary_loss = torch.tensor(weight).to(DEVICE) * \
                                       loss(new_flow, mask = gained_confidence, ground_truth=gt_flows, device=DEVICE)
                        train_total_loss = train_total_loss + unitary_loss

                        # normalize loss by the number of videos in the test dataset and the bunch of epochs
                        train_loss2print[i] += unitary_loss.item() / (len(train_loader) * param.SHOWING_N_ITER)

                        i += 1

                    train_total_loss.backward()
                    optimizer.step()

                    # mask update before next step
                    confidence = confidence_new.clone().detach()


        #Print statistics
        if (epoch % param.SHOWING_N_ITER == 0) and (TB_writer is not None):
            print(' [Epoch for Update %5d]' % epoch)

            # TODO: Moverlo a una función junto a lo mismo que está en el training
            verbose_dir = join(param.VERBOSE_ROOT_DIR, param.TRAINING_NAME)
            update_net.eval()

            ## Print Train Losses
            scalars_to_show = train_loss2print + [sum(train_loss2print)]  # + concatenates
            name_of_scalars = train_loss_names + ['Update Total Train Loss']
            show_statistics(epoch, verbose_dir, scalars_to_show, name_of_scalars, '', TB_writer)
            # Prepare for the next bunch of epochs
            train_loss2print = [0] * len(losses['losses_list'])


            # Print Test Losses
            # each component corresponds to a loss and it is the sum of the loses of all videos for the given loss
            test_loss = [0] * len(losses['losses_list'])

            # each component correspond to a video in the dataset
            flows_to_print = []
            computed_flows_to_print = []
            computed_frames_to_print = []

            #update_net.eval()
            with torch.no_grad():
                for frames, flows, masks, gt_frames, gt_flows in test_loader:
                    # Remove the batch dimension (for pierrick architecture is needed B to be 1)
                    B, N, C, H, W = flows.shape

                    frames = frames.view(B*N, 3, H, W)
                    flows = flows.view(B * N, C, H, W)
                    # masks: 1 inside the hole
                    masks = masks.view(B * N, 1, H, W)
                    gt_frames = gt_frames.view(B*N, 3, H, W)
                    gt_flows = gt_flows.view(B * N, C, H, W)

                    # place data on device
                    flows = flows.to(DEVICE)
                    masks = masks.to(DEVICE)
                    gt_flows = gt_flows.to(DEVICE)


                    # Initial confidence: 1 outside the mask (the hole), 0 inside
                    initial_confidence = 1 - 1. * masks
                    confidence = initial_confidence.clone()
                    gained_confidence = initial_confidence

                    new_flow = flows.clone()
                    F = flow2F(flows)

                    step = -1
                    loss_to_print=[]
                    while (gained_confidence.sum() > 0) and (step <= param.max_num_steps):
                        step += 1
                        #print('Test step: ', str(step))

                        current_flow = new_flow.clone().detach()
                        current_F = F.clone().detach()
                        
                        new_F, confidence_new = update_step(update_net, current_flow, current_F, confidence)
                        gained_confidence = (confidence_new > confidence) * confidence_new

                        if gained_confidence.sum() > 0:
                            F = current_F * (confidence_new <= confidence) + new_F * (confidence_new > confidence)
                            new_flow = F2flow(F)

                            i = 0
                            for loss, weight in zip(losses['losses_list'], losses['weights_list']):
                                unitary_loss = weight * loss(new_flow, mask=gained_confidence, ground_truth=gt_flows,
                                                             device=DEVICE).item()

                                test_loss[i] += unitary_loss / len(test_loader)

                                i += 1

                        # mask update before next step
                        confidence = confidence_new.clone().detach()

                        #Print masks
                        folder = join(verbose_dir, 'masks')
                        create_dir(folder)
                        m_np = confidence.cpu().numpy()
                        for n_frame in range(confidence.shape[0]):
                            m_pil = Image.fromarray(255 * np.squeeze(m_np[n_frame, :, :]))
                            if m_pil.mode != 'RGB':
                                m_pil = m_pil.convert('RGB')
                            m_pil.save(folder + '/{:04d}_{:02d}.png'.format(n_frame, step))

                    flows_to_print.append(flows)
                    computed_flows_to_print.append(new_flow)

                    # computed_frames = from_flow_to_frame_seamless(frames=frames, flows=flows, masks=masks)
                    computed_frames = from_flow_to_frame(frames=frames, flows=flows, masks=masks)
                    computed_frames_to_print.append(computed_frames)



            scalars_to_show = test_loss + [sum(test_loss) ] # + concatenates
            name_of_scalars =  test_loss_names + ['Update Total Test Loss']

            show_statistics(epoch, verbose_dir, scalars_to_show, name_of_scalars, 'update_', TB_writer,
                            input_flow_list=flows_to_print, computed_flow_list=computed_flows_to_print,
                            computed_frames_list=computed_frames_to_print)

            update_net.train()

        if (epoch % param.SAVING_N_ITER == 0) and (chk_path is not None):
            # save checkpoint
            chk = {
                'epoch': epoch,
                'enc_state_dict': flow2F.state_dict(),
                'dec_state_dict': F2flow.state_dict(),
                'update_state_dict': update_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(chk, chk_path)


# TODO: Funcion que debe ir a algun fichero de utilidades
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def warp(features, field, device):
    # features: size (CxHxW)
    # field: size (2xHxW)

    C, H, W = features.shape

    # Grid for warping
    xx, yy = torch.meshgrid(torch.arange(H), torch.arange(W))
    ind = torch.stack((yy, xx), dim=-1).to(device)

    field = field.permute((1, 2, 0)) + ind
    field = torch.unsqueeze(field, 0)

    # Normalize the coordinates to the square [-1,1]
    field = (2 * field / torch.tensor([W, H]).view(1, 1, 1, 2).to(device)) - 1

    # warp ## FORWARD ##
    features2warp = torch.unsqueeze(features, 0)
    warped_features = torch.nn.functional.grid_sample(features2warp, field,
                                          mode='bilinear', padding_mode='border',
                                          align_corners=False)
    warped_features = torch.squeeze(warped_features)

    return warped_features

def update_step(update, flow, F, confidence):

    N, C, H, W = flow.shape

    new_F = F * 0
    confidence_new = confidence.clone()

    for n_frame in range(N):
        frame_flow = flow[n_frame]

        ## warping
        if n_frame + 1 < N:
            F_f = warp(F[n_frame + 1, :, :, :], frame_flow[:2, :, :], DEVICE)
            confidence_f = warp(confidence[n_frame + 1, :, :, :], frame_flow[:2, :, :],
                                device=DEVICE)
        else:
            F_f = 0. * F[n_frame]
            confidence_f = 0. * confidence[n_frame]

        if n_frame - 1 >= 0:
            F_b = warp(F[n_frame - 1, :, :, :], frame_flow[2:], device=DEVICE)
            confidence_b = warp(confidence[n_frame - 1, :, :, :], frame_flow[2:], device=DEVICE)
        else:
            F_b = 0. * F[n_frame]
            confidence_b = 0. * confidence[n_frame]
        # End warping

        # input of the update network is the concatenation of the obtained features from this frame and the neighboring ones
        x = torch.cat((F_b, F[n_frame], F_f), dim=0)

        confidence_in = torch.cat(((confidence_b).repeat(F_b.shape[0], 1, 1),
                                   confidence[n_frame].repeat(F[n_frame].shape[0], 1, 1),
                                   (confidence_f).repeat(F_f.shape[0], 1, 1)),
                                  dim=0)  # same goes for the input mask

        # free memry as much as posible


        ### UPDATE ###
        new_F[n_frame], confidence_new[n_frame] = update(x, confidence_in)  # Update



        # force the initially confident pixels to stay confident, because a decay can be observed
        # depending on the update rule of the partial convolution
        confidence_new[n_frame][confidence[n_frame] == 1] = 1.


    return new_F, confidence_new


def main():
    #Folders
    TB_stats_dir = join(param.TB_STATS_ROOT_DIR, param.TRAINING_NAME)
    checkpoint_dir = join(param.CHECKPOINT_ROOT_DIR, param.TRAINING_NAME)

    #Setup the Tensor Board stuff for statistics
    TB_writer = SummaryWriter(TB_stats_dir)


    ## DATAASETS
    #Dataset for Encoder/Decode
    encDec_train_data = VideoInp_DataSet(param.ENC_DEC_TRAIN_ROOT_DIR, training=True, random_mask_on_the_fly=param.encdDec_random_mask_on_the_fly)
    encDec_train_loader = DataLoader(encDec_train_data, batch_size=1, shuffle=True, drop_last=False)

    encDec_test_data = VideoInp_DataSet(param.ENC_DEC_TEST_ROOT_DIR, training=True, random_mask_on_the_fly=False)
    encDec_test_loader = DataLoader(encDec_test_data, batch_size=1, shuffle=False, drop_last=False)

    #Dataset for Update
    update_train_data = VideoInp_DataSet(param.UPDATE_TRAIN_ROOT_DIR, training=True,
                                         random_mask_on_the_fly=param.update_random_mask_on_the_fly)
    update_train_loader = DataLoader(update_train_data, batch_size=1, shuffle=True, drop_last=False)

    update_test_data = VideoInp_DataSet(param.UPDATE_TEST_ROOT_DIR, training=True, random_mask_on_the_fly=False)
    update_test_loader = DataLoader(update_test_data, batch_size=1, shuffle=False, drop_last=False)

    # Net Models
    flow2F = Flow2features()
    F2flow = Features2flow()
    update_net = Res_Update(update=param.partial_mode_update)

    # Optimizers
    encDec_optimizer = optim.Adam(list(F2flow.parameters()) + list(flow2F.parameters()),
                                   lr=param.adam_lr,
                                   betas=param.adam_betas,
                                   weight_decay=param.adam_weight_decay)

    update_optimizer = optim.Adam(update_net.parameters(),
                           lr=param.adam_lr,
                           betas=param.adam_betas,
                           weight_decay=param.adam_weight_decay)

    #If exists checkpoints, load it
    create_dir(checkpoint_dir)
    encDec_checkpoint_filename = join(checkpoint_dir, param.ENC_DEC_CHECKPOINT_FILENAME)
    update_checkpoint_filename = join(checkpoint_dir, param.UPDATE_CHECKPOINT_FILENAME)

    encDec_epoch = 0
    if os.path.exists(encDec_checkpoint_filename):
        checkpoint = torch.load(encDec_checkpoint_filename, map_location='cpu')

        flow2F.load_state_dict(checkpoint['enc_state_dict'])
        F2flow.load_state_dict(checkpoint['dec_state_dict'])
        encDec_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        encDec_epoch = checkpoint['epoch']

        print('** Weights ' + encDec_checkpoint_filename + ' loaded \n')

    update_epoch = 1
    if os.path.exists(update_checkpoint_filename):
        checkpoint = torch.load(update_checkpoint_filename, map_location='cpu')
        update_net.load_state_dict(checkpoint['update_state_dict'])
        update_optimizer.load_state_dict((checkpoint['optimizer_state_dict']))
        update_epoch = checkpoint['epoch']

        print('** Weights ' + update_checkpoint_filename + ' loaded \n')


    #Move things to the available device
    flow2F.to(DEVICE)
    F2flow.to(DEVICE)
    update_net.to(DEVICE)
    optimizer_to(encDec_optimizer, DEVICE)
    optimizer_to(update_optimizer, DEVICE)

    #####################################
    # Encoder Decoder Training
    flow2F.train()
    F2flow.train()
    torch.autograd.set_detect_anomaly(True)

    train_encoder_decoder(flow2F, F2flow, encDec_train_loader, encDec_test_loader, encDec_optimizer, param.encDec_losses, final_epoch=param.encDec_n_epochs,
                          ini_epoch=encDec_epoch, TB_writer=TB_writer, chk_path=encDec_checkpoint_filename)

    print('Encoder-decoder already trained with ', str(encDec_epoch), ' epochs')
    #############

    ###################################
    # Update Training

    # freeze encoder/decoder
    for learn_param in flow2F.parameters():
        learn_param.requires_grad = False

    for learn_param in F2flow.parameters():
        learn_param.requires_grad = False

    flow2F.eval()
    F2flow.eval()
    update_net.train()
    train_update(flow2F, F2flow, update_net, update_train_loader, update_test_loader, update_optimizer, param.update_losses, final_epoch=param.update_n_epochs,
              ini_epoch=update_epoch, TB_writer=TB_writer, chk_path=update_checkpoint_filename)

    print('Update already trained with ', str(encDec_epoch), ' epochs')

if __name__ == '__main__':
    main()
