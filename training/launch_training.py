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

########################################################################################################################
# PARAMETERS








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

def show_statistics(iter, verbose_dir, metrics_to_show, titles, caption, input_flow, computed_flow, gt_flow, writer):
    print(caption + ' [Epoch %5d]' % iter)

    for metric, title in zip(metrics_to_show, titles):
        writer.add_scalar(title, metric, iter)
        print('\t\t' + title +'  : %.3f' % (metric))

    #plot_optical_flow(computed_flow, writer, caption + " : Computed Flows")
    #plot_optical_flow(gt_flow, writer, caption + " : Ground Truth Flows")


    # save Forward flow images
    folder = join(verbose_dir + caption, 'input_forward_flow')
    tensor_save_flow_and_img(input_flow[:, 0:2, :, :], folder)

    folder = join(verbose_dir + caption, 'computed_forward_flow')
    tensor_save_flow_and_img(computed_flow[:, 0:2, :, :], folder)

    folder = join(verbose_dir + caption, 'GT_forward_flow')
    tensor_save_flow_and_img(gt_flow[:, 0:2, :, :], folder)

    # save Backward flow images
    folder = join(verbose_dir + caption, 'input_backward_flow')
    tensor_save_flow_and_img(input_flow[:, 2:, :, :], folder)

    folder = join(verbose_dir + caption, 'computed_backward_flow')
    tensor_save_flow_and_img(computed_flow[:, 2:, :, :], folder)

    folder = join(verbose_dir + caption, 'GT_backward_flow')
    tensor_save_flow_and_img(gt_flow[:, 2:, :, :], folder)

#############################

def train_all(flow2F, F2flow, update_net, train_loader, test_loader,
              optimizer, encDec_losses, update_losses,
              n_epochs=100, ini_epoch=1, TB_writer =None, chk_path=None):
    mu = 0
    loss_encDec_print = 0
    loss_update_print = 0
    total_loss_print = 0
    for epoch in range(ini_epoch, ini_epoch + n_epochs+1):
        for i, (frames, iflows, masks, gt_frames, gt_flows) in enumerate(train_loader):
            # Remove the batch dimension (for pierrick architecture is needed B to be 1)
            B, N, C, H, W = iflows.shape

            frames = frames.view(B*N, 3, H, W)
            iflows = iflows.view(B * N, C, H, W)
            # masks: 1 inside the hole
            masks = masks.view(B * N, 1, H, W)

            gt_frames = gt_frames.view(B*N, 3, H, W)
            gt_flows = gt_flows.view(B * N, C, H, W)

            # place data on device
            iflows = iflows.to(DEVICE)
            masks = masks.to(DEVICE)
            gt_flows = gt_flows.to(DEVICE)

            # Initial confidence: 1 outside the mask (the hole), 0 inside
            initial_confidence = 1 - masks
            confidence = initial_confidence.clone()
            confidence_new = initial_confidence * 0

            new_flow = iflows.clone()

            step = -1
            while (1-confidence_new).sum() >0:
                step += 1

                optimizer.zero_grad()

                current_flow = new_flow.clone().detach()
                F = flow2F(current_flow)
                encDec_flow = F2flow(F)

                loss_encDec = 0
                for loss, weight in zip(encDec_losses['losses_list'], encDec_losses['weights_list']):
                    loss_encDec += torch.tensor(weight).to(DEVICE) * \
                                   loss(encDec_flow, ground_truth=gt_flows)

                new_F, confidence_new = update_step(update_net, current_flow, F, confidence, initial_confidence)

                gained_confidence = (confidence_new > confidence) * confidence_new
                F = F * (confidence_new <= confidence) + new_F * (confidence_new > confidence)

                new_flow = F2flow(F)

                loss_update = 0
                if gained_confidence.sum() != 0:
                    for loss, weight in zip(update_losses['losses_list'], update_losses['weights_list']):
                        loss_update += torch.tensor(weight).to(DEVICE) * \
                                       loss(new_flow, mask = gained_confidence, ground_truth=gt_flows, device=DEVICE)

                mu = torch.tensor([1 - np.exp(-epoch / param.S_0)]).to(DEVICE)
                loss_update = mu * loss_update

                total_loss = ( loss_encDec + mu * (loss_update ))

                total_loss.backward()  # weighting of the loss
                optimizer.step()

                # mask update before next step
                confidence = confidence_new*1

        #Print statistics
        if (epoch % param.SHOWING_N_ITER == 0) and (TB_writer is not None):
            # TEST TODO: Moverlo a una función junto a lo mismo que está en el training
            verbose_dir = join(param.VERBOSE_ROOT_DIR, param.TRAINING_NAME)
            flow2F.eval()
            F2flow.eval()
            update_net.eval()

            with torch.no_grad():
                for i, (frames, iflows, masks, gt_frames, gt_flows) in enumerate(test_loader):
                    # Remove the batch dimension (for pierrick architecture is needed B to be 1)
                    B, N, C, H, W = iflows.shape

                    frames = frames.view(B*N, 3, H, W)
                    iflows = iflows.view(B * N, C, H, W)
                    # masks: 1 inside the hole
                    masks = masks.view(B * N, 1, H, W)
                    gt_frames = gt_frames.view(B*N, 3, H, W)
                    gt_flows = gt_flows.view(B * N, C, H, W)

                    # place data on device
                    iflows = iflows.to(DEVICE)
                    masks = masks.to(DEVICE)
                    gt_flows = gt_flows.to(DEVICE)

                    # Initial confidence: 1 outside the mask (the hole), 0 inside
                    initial_confidence = 1 - masks
                    confidence = initial_confidence

                    flow = iflows

                    step = -1
                    loss_to_print=[]
                    while (1 - confidence).sum() > 0:
                        step += 1

                        F = flow2F(flow)

                        new_F, confidence_new = update_step(update_net, flow, F, confidence, initial_confidence)

                        gained_confidence = (confidence_new > confidence) * confidence_new
                        F = F * (confidence_new <= confidence) + new_F * (confidence_new > confidence)

                        flow = F2flow(F)

                        # mask update before next step
                        confidence = confidence_new * 1.

                        #Print masks
                        folder = join(verbose_dir, 'masks')
                        create_dir(folder)
                        m_np = confidence.cpu().numpy()
                        for n_frame in range(confidence.shape[0]):
                            m_pil = Image.fromarray(255 * np.squeeze(m_np[n_frame, :, :]))
                            if m_pil.mode != 'RGB':
                                m_pil = m_pil.convert('RGB')
                            m_pil.save(folder + '/{:04d}_{:02d}.png'.format(n_frame, step))

                    test_flow = F2flow(F)

                    mu = 1 - np.exp(-epoch / param.S_0)


                    test_loss_encDec = []
                    name_loss_encDec = []
                    for loss, weight in zip(encDec_losses['losses_list'], encDec_losses['weights_list']):
                        test_loss_encDec.append( torch.tensor(weight).to(DEVICE) * \
                                       loss(test_flow, ground_truth = gt_flows))

                        name_loss_encDec.append('Encoder_Decoder Loss : ' + loss.__name__)

                    test_loss_update = []
                    name_loss_update =[]
                    for loss, weight in zip(update_losses['losses_list'], update_losses['weights_list']):
                        test_loss_update.append(torch.tensor(weight).to(DEVICE) * \
                                       loss(test_flow, mask=gained_confidence, ground_truth=gt_flows, device=DEVICE))

                        name_loss_update.append('Update_Loss : ' + loss.__name__)

                    test_total_loss = sum(test_loss_encDec) + mu * sum(test_loss_update)

                    #video = from_flow_to_frame_seamless(frames=frames, flows=flow, masks=masks)
                    video = from_flow_to_frame(frames=frames, flows=flow, masks=masks)
                    video_gt = gt_frames.clone().detach().cpu().permute(2, 3, 1, 0).numpy()

                    pointwise_error = np.mean(np.abs(video - video_gt), axis=2)
                    frame_error = np.mean(np.mean(pointwise_error, axis=0))

                    #concatenate
                    scalars_to_show = test_loss_encDec + test_loss_update + [test_total_loss] + [frame_error] + [mu]
                    name_of_scalars = name_loss_encDec + name_loss_update + ['Total Loss'] + ['Frame Difference'] + ['mu']
                    show_statistics(epoch, verbose_dir, scalars_to_show, name_of_scalars, '', iflows, flow, gt_flows, TB_writer)


                    # print video
                    folder = join(verbose_dir, 'warped_frames')
                    create_dir(folder)
                    for n_frame in range(video.shape[3]):
                        frame_blend = video[:,:,:,n_frame]
                        m_pil = Image.fromarray((255 * np.squeeze(frame_blend)).astype(np.uint8))
                        if m_pil.mode != 'RGB':
                            m_pil = m_pil.convert('RGB')
                        m_pil.save(folder + '/{:04d}_.png'.format(n_frame))

                flow2F.train()
                F2flow.train()
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

def update_step(update,  flow, F, confidence, initial_confidence):

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
        del F_b
        del F_f
        del frame_flow

        ### UPDATE ###
        new_F[n_frame], confidence_new[n_frame] = update(x, confidence_in)  # Update

        del x
        del confidence_in

        # force the initially confident pixels to stay confident, because a decay can be observed
        # depending on the update rule of the partial convolution
        confidence_new[n_frame][initial_confidence[n_frame] == 1] = 1.





    return new_F, confidence_new


def main():
    #Folders
    TB_stats_dir = join(param.TB_STATS_ROOT_DIR, param.TRAINING_NAME)
    checkpoint_dir = join(param.CHECKPOINT_ROOT_DIR, param.TRAINING_NAME)

    #Setup the Tensor Board stuff for statistics

    TB_writer = SummaryWriter(TB_stats_dir)

    train_data = VideoInp_DataSet(param.TRAIN_ROOT_DIR, training=True, random_mask_on_the_fly=param.random_mask_on_the_fly)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, drop_last=False)

    test_data = VideoInp_DataSet(param.TEST_ROOT_DIR, training=True, random_mask_on_the_fly=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, drop_last=False)

    # Net Models
    flow2F = Flow2features()
    F2flow = Features2flow()
    update_net = Res_Update()

    # Optimizer
    optimizer = optim.Adam(list(F2flow.parameters()) + list(flow2F.parameters()) + list(update_net.parameters()),
                           lr=param.adam_lr,
                           betas=param.adam_betas,
                           weight_decay=param.adam_weight_decay)

    #If exists checkpoint, load it
    create_dir(checkpoint_dir)
    checkpoint_filename = join(checkpoint_dir, param.CHECKPOINT_FILENAME)

    epoch = 1
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

    torch.autograd.set_detect_anomaly(True)

    train_all(flow2F, F2flow, update_net, train_loader, test_loader, optimizer, param.encDec_losses, param.update_losses, n_epochs=param.n_epochs,
              ini_epoch=epoch, TB_writer=TB_writer, chk_path=checkpoint_filename)

if __name__ == '__main__':
    main()