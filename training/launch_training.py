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
import torchvision
from PIL import Image
from utils.data_io import create_dir
from model.flow_losses import  mask_L1_loss, L1_loss

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from utils.flow_viz import flow_to_image



from tqdm import tqdm
import numpy as np

from utils.data_io import tensor_save_flow_and_img
from os.path import join
from torchviz import make_dot
# PARAMETERS

root_dir = '../dataset'
TB_STATS_DIR = '../tensor_board'
VERBOSE_DIR ='../training_out'

CHECKPOINT_DIR = '../checkpoint/'
CHECKPOINT_FILENAME = 'all.tar'

S_0 = 1000

SHOW_EACH = 10  # The loss is shown every n_iter

#################

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

def show_statistics(iter, metrics_to_show, titles, pre_caption,  computed_flow, gt_flow, writer):
    print(pre_caption + ' [Epoch %5d]' % iter)
    for metric, title in zip(metrics_to_show, titles):
        writer.add_scalar(title, metric, iter)
        print('\t\t' + title +'  : %.3f' % (metric))

    plot_optical_flow(computed_flow, writer, pre_caption + " : Computed Flows")
    plot_optical_flow(gt_flow, writer, pre_caption + " : Ground Truth Flows")



    # save Forward flow images
    folder = join(VERBOSE_DIR + pre_caption, 'computed_forward_flow')
    tensor_save_flow_and_img(computed_flow[:, 0:2, :, :], folder)

    folder = join(VERBOSE_DIR + pre_caption, 'GT_forward_flow')
    tensor_save_flow_and_img(gt_flow[:, 0:2, :, :], folder)

    # save Backward flow images
    folder = join(VERBOSE_DIR + pre_caption, 'computed_backward_flow')
    tensor_save_flow_and_img(computed_flow[:, 2:, :, :], folder)

    folder = join(VERBOSE_DIR + pre_caption, 'GT_backward_flow')
    tensor_save_flow_and_img(gt_flow[:, 2:, :, :], folder)


#############################

def train_encoder_decoder(encoder, decoder, train_loader, optim, loss_computer, n_epochs=100, ini_epoch=0,
                          TB_writer = None, chk_path=None):
    # initialize loss to print

    for epoch in range(ini_epoch, ini_epoch + n_epochs+1):
        loss2print = 0.0
        for i, data in enumerate(train_loader):
            # get the input, data is a tuple composed by
            flows, mask, gt_flows = data

            B, N, C, H, W = flows.shape

            #Remove the batch dimension (for pierrick architecture is needed B to be 1)
            flows = flows.view(B * N, C, H, W)
            mask = mask.view(B * N, 1, H, W)
            gt_flows = gt_flows.view(B * N, C, H, W)

            # place data on device
            flows = flows.to(DEVICE)
            mask = mask.to(DEVICE)
            gt_flows = gt_flows.to(DEVICE)

            # zero the parameter gradients
            optim.zero_grad()

            # Forward Pass
            computed_flows = decoder(encoder(flows))

            loss = loss_computer(computed_flows, gt_flows)

            # Backward Pass
            loss.backward()
            optim.step()

            #update loss to print
            loss2print += loss.item()

            #print loss tatistics
            show_each = 100 #The loss is shown every n_iter
            if (epoch % show_each == 0) and (TB_writer is not None):
                show_statistics(epoch, loss.item(), 'Encoder_Decoder', flows, computed_flows, gt_flows, TB_writer)
                loss2print = 0.0

            if (epoch % show_each == 0) and (chk_path is not None):
                #save checkpoint
                chk = {
                    'epoch': epoch,
                    'enc_state_dict': encoder.state_dict(),
                    'dec_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'device': DEVICE
                }
                torch.save(chk, chk_path)


def train_all(flow2F, F2flow, update_net, train_loader, optimizer, f_loss_all_pixels, f_mask_loss, n_epochs=100, ini_epoch=1, TB_writer =None, chk_path=None):
    mu = 0
    loss_encDec_print = 0
    loss_update_print = 0
    total_loss_print = 0
    for epoch in range(ini_epoch, ini_epoch + n_epochs+1):
        for i, (iflows, masks, gt_flows) in enumerate(train_loader):
            # Remove the batch dimension (for pierrick architecture is needed B to be 1)
            B, N, C, H, W = iflows.shape
            iflows = iflows.view(B * N, C, H, W)
            # masks: 1 inside the hole
            masks = masks.view(B * N, 1, H, W)
            gt_flows = gt_flows.view(B * N, C, H, W)

            # place data on device
            iflows = iflows.to(DEVICE)
            masks = masks.to(DEVICE)
            gt_flows = gt_flows.to(DEVICE)

            # Initial confidence: 1 outside the mask (the hole), 0 inside
            initial_confidence = 1 - masks
            current_confidence = initial_confidence
            confidence_new = initial_confidence * 0


            #for step in tqdm(range(6), desc='## Step  ##', position=0):


            new_flow = iflows

            # TODO: Sustituir el 6 por algo mas inteligente. Iterar hasta que se hayan visitado todos los píxeles internos
            step = -1
            #for step in range(6):
            while (1-confidence_new).sum() >0:
                step += 1
                optimizer.zero_grad()

                current_flow = new_flow.clone().detach()
                F = flow2F(current_flow)
                encDec_flow = F2flow(F)
                loss_encDec = f_mask_loss(encDec_flow, gt_flows, current_confidence)

                loss_encDec_print += loss_encDec

                new_F = F * 0
                #for n_frame in tqdm(range(N), desc='   Frame', position=1, leave=False):
                for n_frame in range(N):
                    frame_decoded_flow = current_flow[n_frame]

                    with torch.no_grad():
                        ## warping
                        if n_frame + 1 < N:
                            F_f = warp(F[n_frame + 1, :, :, :], frame_decoded_flow[:2, :, :], DEVICE)
                            confidence_f = warp(current_confidence[n_frame + 1, :, :, :], frame_decoded_flow[:2, :, :], device=DEVICE)
                        else:
                            F_f = 0. * F[n_frame]
                            confidence_f = 0. * current_confidence[n_frame]

                        if n_frame - 1 >= 0:
                            F_b = warp(F[n_frame - 1, :, :, :], frame_decoded_flow[2:], device=DEVICE)
                            confidence_b = warp(current_confidence[n_frame - 1, :, :, :], frame_decoded_flow[2:], device=DEVICE)
                        else:
                            F_b = 0. * F[ n_frame]
                            confidence_b = 0. * current_confidence[n_frame]
                        # End warping

                        # input of the update network is the concatenation of the obtained features from this frame and the neighboring ones
                        x = torch.cat((F_b, F[n_frame], F_f), dim=0)

                        confidence_in = torch.cat(((confidence_b).repeat(F_b.shape[0], 1, 1),
                                                   current_confidence[n_frame].repeat(F[n_frame].shape[0], 1, 1),
                                                   (confidence_f).repeat(F_f.shape[0], 1, 1)),
                                                  dim=0)  # same goes for the input mask

                        # free memry as much as posible
                        del F_b
                        del F_f
                        del frame_decoded_flow

                    ### UPDATE ###
                    new_F[ n_frame], confidence_new[n_frame] = update_net(x, confidence_in)  # Update

                    del x
                    del confidence_in

                    # force the initially confident pixels to stay confident, because a decay can be observed
                    # depending on the update rule of the partial convolution
                    confidence_new[ n_frame][initial_confidence[ n_frame] == 1] = 1.

                    #Print Results
                    '''
                    folder = join('salida_entreno_all', 'mask')
                    create_dir(folder)
                    m_np = confidence_new.cpu().numpy()
                    m_pil = Image.fromarray(255*np.squeeze(m_np[n_frame,:,:]))
                    if m_pil.mode != 'RGB':
                        m_pil = m_pil.convert('RGB')
                    m_pil.save(folder + '/{:04d}_{:02d}.png'.format(n_frame, step))
                    '''

                gained_confidence = (confidence_new > current_confidence) * confidence_new
                F = F * (confidence_new <= current_confidence) + new_F * (confidence_new > current_confidence)

                new_flow = F2flow(F)
                if gained_confidence.sum != 0:
                    loss_update = f_mask_loss(new_flow, gt_flows, gained_confidence)

                    current_confidence = confidence_new * 1.  # mask update before next step

                    mu = 1 - np.exp(-epoch / S_0)

                    total_loss = (1. * loss_encDec + mu * loss_update)
                    loss_update_print += loss_update
                    total_loss_print += total_loss


                    total_loss.backward()  # weighting of the loss
                    optimizer.step()



                del current_flow

            # print loss tatistics

            if (epoch % SHOW_EACH == 0) and (TB_writer is not None):
                show_statistics(epoch ,
                                [loss_encDec_print.item(), loss_update_print.item(), total_loss_print.item(), mu],
                                ['Encoder/Decoder Loss', 'Update Loss', 'Total loss', 'Mu'], '', new_flow, gt_flows, TB_writer)
                loss_encDec_print = 0
                loss_update_print = 0
                total_loss_print = 0


            if (epoch % SHOW_EACH == 0) and (chk_path is not None):
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
    # features: size (CxWxH)
    # field: size (2xWxH)

    C, H, W = features.shape

    # Grid for warping
    xx, yy = torch.meshgrid(torch.arange(H), torch.arange(W))
    ind = torch.stack((yy, xx), dim=-1).to(device)

    field = field.permute((1, 2, 0)) + ind
    field = torch.unsqueeze(field, 0)

    # Normalize the coordinates to the square [-1,1]
    field = (2 * field / torch.tensor([W, H]).view(1, 1, 1, 2).to(DEVICE)) - 1

    # warp ## FORWARD ##
    features2warp = torch.unsqueeze(features, 0)
    warped_features = torch.nn.functional.grid_sample(features2warp, field,
                                          mode='bilinear', padding_mode='border',
                                          align_corners=False)
    warped_features = torch.squeeze(warped_features)

    return warped_features


if __name__ == '__main__':
    #Setup the Tensor Board stuff for statistics
    TB_writer = SummaryWriter(TB_STATS_DIR)

    train_data = VideoInp_DataSet(root_dir)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, drop_last=True)

    # Net Models
    flow2F = Flow2features()
    F2flow = Features2flow()
    update_net = Res_Update()

    # Optimazer
    optimizer = optim.Adam(list(F2flow.parameters()) + list(flow2F.parameters()) + list(update_net.parameters()),
                           lr=1e-4,
                           betas=(0.9, 0.999),
                           weight_decay=0.00004)

    #If exists checkpoint, load it
    create_dir(CHECKPOINT_DIR)


    checkpoint_filename = join(CHECKPOINT_DIR, CHECKPOINT_FILENAME)

    epoch=1
    if os.path.exists(checkpoint_filename):
        checkpoint = torch.load(checkpoint_filename, map_location='cpu')

        flow2F.load_state_dict(checkpoint['enc_state_dict'])
        F2flow.load_state_dict(checkpoint['dec_state_dict'])
        update_net.load_state_dict(checkpoint['update_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

        print('** Weights ' + checkpoint_filename + ' loaded \n')


    flow2F.to(DEVICE)
    F2flow.to(DEVICE)
    update_net.to(DEVICE)
    optimizer_to(optimizer, DEVICE)

    flow2F.train()
    F2flow.train()
    update_net.train()

    #loss
    loss_encDec = L1_loss
    loss_update = mask_L1_loss

    torch.autograd.set_detect_anomaly(True)


    train_all(flow2F, F2flow, update_net, train_loader, optimizer, loss_encDec, loss_update, 300000,
              ini_epoch=epoch, TB_writer=TB_writer, chk_path=checkpoint_filename)

