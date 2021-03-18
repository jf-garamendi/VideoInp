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

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from utils.flow_viz import flow_to_image



from tqdm import tqdm
import numpy as np

from utils.data_io import tensor_save_flow_and_img
from os.path import join

# TODO: Argumentos a linea de comandos
root_dir = '../data_t'
TB_STATS_DIR = './tensor_board'
ENC_DEC_CHECKPOINT_DIR = './checkpoint/'
UPDATE_CHECKPOINT_DIR = './checkpoint/'

ENC_DEC_CHECKPOINT_FILENAME = 'chk_enc_dec.tar'
UPDATE_CHECKPOINT_FILENAME = 'chk_update.tar'


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

def show_statistics(iter, training_loss, pre_caption, input_flow, computed_flow, gt_flow, writer):
    writer.add_scalar(pre_caption +' : Training Loss', training_loss, iter)

    plot_optical_flow(input_flow, writer, pre_caption + " : Input Flows")
    plot_optical_flow(computed_flow, writer, pre_caption + " : Computed Flows")
    plot_optical_flow(gt_flow, writer, pre_caption + " : Ground Truth Flows")

    print(pre_caption + ' [Epoch %5d] loss: %.3f' % (iter,  training_loss))

    # save Forward flow images
    folder = join('salida_entreno_' + pre_caption, 'computed_forward_flow')
    tensor_save_flow_and_img(computed_flow[:, 0:2, :, :], folder)

    folder = join('salida_entreno_' + pre_caption, 'GT_forward_flow')
    tensor_save_flow_and_img(gt_flow[:, 0:2, :, :], folder)

    folder = join('salida_entreno_' + pre_caption, 'input_forward_flow')
    tensor_save_flow_and_img(input_flow[:, 0:2, :, :], folder)

    # save Backward flow images
    folder = join('salida_entreno_' + pre_caption, 'computed_backward_flow')
    tensor_save_flow_and_img(computed_flow[:, 2:, :, :], folder)

    folder = join('salida_entreno_' + pre_caption, 'GT_backward_flow')
    tensor_save_flow_and_img(gt_flow[:, 2:, :, :], folder)

    folder = join('salida_entreno_' + pre_caption, 'input_backward_flow')
    tensor_save_flow_and_img(input_flow[:, 2:, :, :], folder)

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
            show_each = 10 #The loss is shown every n_iter
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





def train_all(flow2F, F2flow, update_net, train_loader, optimizer, loss_computer, n_epochs=100, ini_epoch=0, TB_writer =None, chk_path=None):
    for epoch in range(ini_epoch, ini_epoch + n_epochs+1):
        it = 0
        #for i, (flows, masks, gt_flows) in enumerate(tqdm(train_loader)):
        for i, (flows, masks, gt_flows) in enumerate(train_loader):
            it += 1

            # Remove the batch dimension (for pierrick architecture is needed B to be 1)
            B, N, C, H, W = flows.shape
            flows = flows.view(B * N, C, H, W)
            masks = masks.view(B * N, 1, H, W)
            gt_flows = gt_flows.view(B * N, C, H, W)

            # place data on device
            flows = flows.to(DEVICE)
            masks = masks.to(DEVICE)
            gt_flows = gt_flows.to(DEVICE)

            # Initial confidence: 1 outside the mask, 0 inside
            initial_confidence = 1 - masks
            current_confidence = initial_confidence

            # Forward pass for features
            F = flow2F(flows)

            # Initialize losses
            loss_1=0
            loss_2=0
            #for step in tqdm(range(6), desc='## Step  ##', position=0):
            for step in range(6):
                all_frames_flow_from_features = F2flow(F)

                confidence_new = current_confidence + 0

                new_F = F * 0
                #for n_frame in tqdm(range(N), desc='   Frame', position=1, leave=False):
                for n_frame in range(N):
                    decoded_flows = all_frames_flow_from_features[n_frame]

                    ## warping
                    if n_frame + 1 < N:
                        field = decoded_flows[:2, :, :]

                        F_f = warp(F[n_frame + 1, :, :, :], field, DEVICE)
                        confidence_f = warp(current_confidence[n_frame + 1, :, :, :], field, device=DEVICE)

                    else:
                        F_f = 0. * F[n_frame]
                        confidence_f = 0. * current_confidence[n_frame]

                    if n_frame - 1 >= 0:
                        field = decoded_flows[2:]

                        F_b = warp(F[n_frame - 1, :, :, :], field, device=DEVICE)
                        confidence_b = warp(current_confidence[n_frame - 1, :, :, :], field, device=DEVICE)

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

                    ### UPDATE ###
                    new_F[ n_frame], confidence_new[n_frame] = update_net(x, confidence_in)  # Update

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

                decoded_flows = F2flow(new_F)

                pointwise_l1_error = torch.abs(gt_flows - decoded_flows).mean(dim=1)

                loss_1 += pointwise_l1_error[torch.squeeze(current_confidence,dim=1) == 1].mean()

                decoded_flows = decoded_flows

                # loss2 is the sum of terms from each step of the iterative scheme
                # we compute loss2 only on the pixels that gained confidence and we weight the result with the new confidence
                s = ((confidence_new > current_confidence) * confidence_new).sum()

                if s != 0:
                    loss_2 += (1 - np.exp(-it / (1000 * (step + 1)))) * (
                            pointwise_l1_error * ((confidence_new > current_confidence) * confidence_new)).sum() / s
                    # we add the weight  (1-np.exp(-it/(1000*(step+1)))) that makes loss2 be 0 for the first batch iteration, then around 1000, we account for the first step of the iterative scheme, and every 1000, another step is smoothly taken into account.
                    # justification: in the beginning, we want the translation between fows and features to converge before using the predicted flows to update the features. Alos, since the update is recurrent, we want to train well the first steps of it before adding a new one


                    # we handcraft the new feature volume
                else:
                    loss_2 += loss_1 * 0

                F = F * (confidence_new <= current_confidence) + new_F * (confidence_new > current_confidence)

                current_confidence = confidence_new * 1.  # mask update before next step

            optimizer.zero_grad()
            total_loss = (1. * loss_1 + 1. * loss_2)
            total_loss.backward()  # weighting of the loss
            optimizer.step()
            decoded_flows = decoded_flows *1

            # print loss tatistics
            show_each = 10  # The loss is shown every n_iter
            if (epoch % show_each == 0) and (TB_writer is not None):
                show_statistics(epoch , total_loss.item(), 'Update', flows, decoded_flows, gt_flows, TB_writer)

            if (epoch % show_each == 0) and (chk_path is not None):
                # save checkpoint
                chk = {
                    'epoch': epoch,
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


## Losses TODO: Tienen que ir a un fichero aparte
def mask_L1_loss(flow, gt_flow, mask):
    pointwise_l1_error = (torch.abs(gt_flow - flow) ** 1).mean(dim=1)
    loss_1 = pointwise_l1_error[mask == 1].mean()

    return loss_1

def L1_loss(flow, gt_flow):
    pointwise_l1_error = (torch.abs(gt_flow - flow) ** 1).mean(dim=1)
    loss_1 = pointwise_l1_error.mean()

    return loss_1


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
    enc_dec_optimizer = optim.Adam(list(F2flow.parameters()) + list(flow2F.parameters()),
                           lr=1e-4,
                           betas=(0.9, 0.999),
                           weight_decay=0.00004)

    update_optimizer = optim.Adam(update_net.parameters(),
                           lr=1e-4,
                           betas=(0.9, 0.999),
                           weight_decay=0.00004)

    #If exists checkpoint, load it
    create_dir(ENC_DEC_CHECKPOINT_DIR)
    create_dir(UPDATE_CHECKPOINT_DIR)

    enc_dec_filename = join(ENC_DEC_CHECKPOINT_DIR, ENC_DEC_CHECKPOINT_FILENAME)
    update_filename = join(UPDATE_CHECKPOINT_DIR, UPDATE_CHECKPOINT_FILENAME)

    enc_dec_epoch=0
    if os.path.exists(enc_dec_filename):
        checkpoint = torch.load(enc_dec_filename, map_location='cpu')
        flow2F.load_state_dict(checkpoint['enc_state_dict'])
        F2flow.load_state_dict(checkpoint['dec_state_dict'])
        enc_dec_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        enc_dec_epoch = checkpoint['epoch']

        print('** Weights ' + enc_dec_filename + ' loaded \n')


    update_epoch = 0
    if os.path.exists(update_filename):
        checkpoint = torch.load(update_filename, map_location='cpu')
        update_net.load_state_dict(checkpoint['update_state_dict'])
        update_optimizer.load_state_dict((checkpoint['optimizer_state_dict']))
        update_epoch = checkpoint['epoch']

        print('** Weights ' + update_filename + ' loaded \n')


    flow2F.to(DEVICE)
    F2flow.to(DEVICE)
    update_net.to(DEVICE)
    optimizer_to(enc_dec_optimizer, DEVICE)
    optimizer_to(update_optimizer, DEVICE)


    flow2F.train()
    F2flow.train()
    update_net.train()



    #loss
    loss_computer = L1_loss
    #train_encoder_decoder(flow2F, F2flow, train_loader, optimizer, loss_computer, 240000)
    train_encoder_decoder(flow2F, F2flow, train_loader, enc_dec_optimizer, loss_computer, 200,
                          ini_epoch= enc_dec_epoch, TB_writer=TB_writer, chk_path=enc_dec_filename)

    for param in flow2F.parameters():
        param.requires_grad = False

    for param in F2flow.parameters():
        param.requires_grad = False


    torch.autograd.set_detect_anomaly(True)
    loss_computer = mask_L1_loss
    train_all(flow2F, F2flow, update_net, train_loader, update_optimizer, loss_computer, 300,
              ini_epoch=update_epoch, TB_writer=TB_writer, chk_path=update_filename)

