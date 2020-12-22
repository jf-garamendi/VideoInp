import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))

import copy
import argparse
from torch.autograd import grad as torch_grad
import matplotlib.pyplot as plt
import torch
import cvbase as cvb
import numpy as np
import cv2
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from torch.utils.data import DataLoader
import iio
import pdb


from models import *
from utils.image import *
from utils.Laplacian import Laplacian
from utils.runner_func import *
from utils.full_data_new import *
from utils.losses import *
from utils.io import *
parser = argparse.ArgumentParser()

Lap=Laplacian(3)

# training options
parser.add_argument('--net', type=str, default='unet')
parser.add_argument('--save_dir', type=str, default='./snapshots/default')
parser.add_argument('--log_dir', type=str, default='./logs/default')
parser.add_argument('--model_name', type=str, default=None)

parser.add_argument('--n_epochs', type=int, default=5)
parser.add_argument('--max_iter', type=int, default=100000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_threads', type=int, default=32)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--get_mask', action='store_true')

parser.add_argument('--LR', type=float, default=1e-4)
parser.add_argument('--LAMBDA_SMOOTH', type=float, default=0.1)
parser.add_argument('--LAMBDA_HARD', type=float, default=2.)
parser.add_argument('--BETA1', type=float, default=0.9)
parser.add_argument('--BETA2', type=float, default=0.999)
parser.add_argument('--WEIGHT_DECAY', type=float, default=0.00004)

parser.add_argument('--IMAGE_SHAPE', type=list, default=[240,424, 3])#240, 424, 3
parser.add_argument('--RES_SHAPE', type=list, default=[240,424, 3])#480 854
parser.add_argument('--FIX_MASK', action='store_true')
parser.add_argument('--MASK_MODE', type=str, default='bbox')
parser.add_argument('--PRETRAINED', action='store_true')
parser.add_argument('--PRETRAINED_MODEL', type=str, default=None)
parser.add_argument('--RESNET_PRETRAIN_MODEL', type=str,
                    default=None)#'pretrained_models/resnet50-19c8e357.pth'
parser.add_argument('--TRAIN_LIST', type=str, default='/home/pierrick/Documents/Projects/UNet/full_list_train')
parser.add_argument('--EVAL_LIST', type=str, default='/home/pierrick/Documents/Projects/UNet/full_list_val')
parser.add_argument('--TEST_LIST', type=str, default='/home/pierrick/Documents/Projects/UNet/full_list_test')
parser.add_argument('--MASK_ROOT', type=str, default=None)
parser.add_argument('--DATA_ROOT', type=str, default=None,
                    help='Set the path to flow dataset')
parser.add_argument('--INITIAL_HOLE', action='store_true')

parser.add_argument('--PRINT_EVERY', type=int, default=5)
parser.add_argument('--MODEL_SAVE_STEP', type=int, default=20000)
parser.add_argument('--NUM_ITERS_DECAY', type=int, default=1000)

parser.add_argument('--MASK_HEIGHT', type=int, default=120)
parser.add_argument('--MASK_WIDTH', type=int, default=212)
parser.add_argument('--VERTICAL_MARGIN', type=int, default=10)
parser.add_argument('--HORIZONTAL_MARGIN', type=int, default=10)
parser.add_argument('--MAX_DELTA_HEIGHT', type=int, default=100)#60 --MAX_DELTA_HEIGHT 40 --MAX_DELTA_WIDTH 40
parser.add_argument('--MAX_DELTA_WIDTH', type=int, default=180)#106

parser.add_argument('--lambda_gan', type=float, default=1/100)
parser.add_argument('--lambda_fb', type=float, default=.1)
parser.add_argument('--lambda_ld', type=float, default=.1)
parser.add_argument('--lambda_consistency', type=float, default=.1)

parser.add_argument('--seg', type=str, default=None)
parser.add_argument('--edge', type=str, default=None)
parser.add_argument('--image', action='store_true')
parser.add_argument('--rflow', action='store_true')

parser.add_argument('--joint', action='store_true') #no additionnal input, 
parser.add_argument('--loss', type=str, default='epe')
parser.add_argument('--patchGAN', action='store_true')
parser.add_argument('--GAN', action='store_true')
parser.add_argument('--multipatchGAN', action='store_true')
parser.add_argument('--patchGANmask', action='store_true')
parser.add_argument('--WpatchGANmask', action='store_true')
parser.add_argument('--in_c', type=int, default=4)
parser.add_argument('--out_c', type=int, default=8)
parser.add_argument('--in_d', type=int, default=30)
parser.add_argument('--n_frames', type=int, default=5)
parser.add_argument('--n_critic', type=int, default=5)
parser.add_argument('--two_masks', action='store_true')
parser.add_argument('--bias_edge', action='store_true')
parser.add_argument('--bbox', action='store_true')
parser.add_argument('--n_steps', type=int, default=5)
parser.add_argument('--update', type=str, default='pow')
parser.add_argument('--schedule', action='store_true')
parser.add_argument('--sandwich', action='store_true')


args = parser.parse_args()

def main():
    flow2F=Flow2features()
    F2flow_dict={}
    param_list=[]
   
    for i in range(1):#args.n_steps):
        F2flow_dict[i]=Features2flow().cuda().train()
        param_list=param_list+list(F2flow_dict[i].parameters())

    net=Res_Update3(update=args.update)

    
    if args.model_name is not None:
        model_save_dir = './snapshots/'+args.model_name+'/ckpt/'
        sample_dir = './snapshots/'+args.model_name+'/images/'
        log_dir = './logs_iterative/'+args.model_name

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir=log_dir)
    torch.cuda.manual_seed(7777777)

    train_dataset = dataset(args,val=False)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=args.n_threads)
    # val_dataset = dataset(args,val=True)
    # val_loader = DataLoader(val_dataset,
    #                           batch_size=args.batch_size,
    #                           shuffle=False,
    #                           drop_last=True,
    #                           num_workers=args.n_threads)
    train_iterator = iter(train_loader)
      
    net.cuda().train()
    flow2F.cuda().train()
    optimizer = optim.Adam(list(net.parameters())+param_list+list(flow2F.parameters()),lr=args.LR,betas=(args.BETA1,args.BETA2),weight_decay=0.*args.WEIGHT_DECAY)
   

    loss = {}
    it = 0 

    for epoch in range(args.n_epochs):
        train_iterator = iter(train_loader)
        print('starting epoch {:d}'.format(epoch))
        for i, (flow_masked,rflow_masked, mask, gt_all_flow, gt_all_rflow, edge, img, bbox) in enumerate(tqdm(train_loader)):
            B,N,C,H,W=flow_masked.shape
            it+=1
            hole=1-mask[:,:,:1].cuda()
            mask_initial=1-1.*mask[:,:,:1].cuda()
            if 'initial_mask' in args.model_name:
                mask_initial=net.initial_mask(mask_initial.view(B*N,1,H,W)).view(B,N,1,H,W)
            mask_current=1.*mask_initial
            gt_all_rflow=gt_all_rflow.cuda()
            gt_all_flow=gt_all_flow.cuda()
            
            
            flow=flow_masked.cuda()
            rflow=rflow_masked.cuda()
            flows=torch.cat((flow,rflow),dim=2)
            gt_flows=torch.cat((gt_all_flow,gt_all_rflow),dim=2)
            
            F=flow2F(flows.view(B*N,2*C,H,W)).view(B,N,32,H,W)
            
            t1,t2=flows.view(B*N,2*C,H,W)[:,:2],flows.view(B*N,2*C,H,W)[:,2:]
            
            xx,yy=torch.meshgrid(torch.arange(H),torch.arange(W))
            ind=torch.stack((yy,xx),dim=-1)
            ind=ind.repeat(B,1,1,1).cuda()
            
            loss_1=0.
            loss_2=0.
            loss_3=0.
            minfbbf=0.
            loss_tv=0.
            total_w=0
            for step in range(args.n_steps):
                
                new_mask=mask_current*0.
                new_F=F*0.
                for frame in range(N):
                    
                    if frame+1<N:
                        
                        grid_f=F2flow_dict[0](F[:,frame])[:,:2].permute((0,2,3,1))+ind #compute forward flow from features
                        grid_f=(2*grid_f/torch.tensor([W,H]).cuda().view(1,1,1,2))-1
                        F_f=torch.nn.functional.grid_sample(F[:,frame+1],grid_f,mode='bilinear',padding_mode='border',align_corners=False)
                        mask_f=torch.clamp(torch.nn.functional.grid_sample(mask_current[:,frame+1],grid_f,mode='bilinear',padding_mode='border',align_corners=False),0,1)
                        #compute interpolation of the features and masks at frame t+1 (if t+1 is a video frame) following the predicted forward flow
                    else:
                        F_f=0.*F[:,frame]
                        mask_f=0.*mask_current[:,frame].cuda()
                 
                    if frame-1>=0:
                        grid_b=F2flow_dict[0](F[:,frame])[:,2:].permute((0,2,3,1))+ind #compute backward flow from features
                        grid_b=(2*grid_b/torch.tensor([W,H]).cuda().view(1,1,1,2))-1
                        F_b=torch.nn.functional.grid_sample(F[:,frame-1],grid_b,mode='bilinear',padding_mode='border',align_corners=False)
                        mask_b=torch.clamp(torch.nn.functional.grid_sample(mask_current[:,frame-1],grid_b,mode='bilinear',padding_mode='border',align_corners=False),0,1)
                    else:
                        F_b=0.*F[:,frame]
                        mask_b=0.*mask_current[:,frame].cuda()
                    
                    x=torch.cat((F_b,F[:,frame],F_f),dim=1)#input of the update network is the concatenation of the obtained features from this frame and the neighboring ones
                    mask_in=torch.clamp(torch.cat(((mask_b).repeat(1,F.shape[2],1,1),mask_current[:,frame].repeat(1,F.shape[2],1,1),(mask_f).repeat(1,F.shape[2],1,1)),dim=1),0,1)#same goes for the input mask
                    new_F[:,frame],new_mask[:,frame]=net(x,mask_in)#Update
                    new_mask[:,frame][hole[:,frame]==1]=1.#force the initially confident pixels to stay confident, because a decay can be observed depending on the update rule of the partial convolution
                    
                # for n,e in locals().items():
                #     if isinstance(e,torch.Tensor):
                #         if torch.isnan(e).sum()!=0:
                #             print(n)
                
                #loss2 is the sum of terms from each step of the iterative scheme
                reconstruction_flows=F2flow_dict[0](new_F.view(B*N,-1,H,W)) 
                pointwise_l1_error=torch.abs(gt_flows.view(B*N,2*C,H,W)-reconstruction_flows).mean(dim=1)
                
                #loss_1+=pointwise_l1_error[hole.view(N*B,H,W)>.2].mean() 
                if args.schedule:
                    s=((1-hole)*new_mask).sum() #we compute loss2 only on the pixels that gained confidence and we weight the result with the new confidence
                    #w=np.exp(-np.abs(it-250*(step+1))**2/(250)**2) if (step<args.n_steps-1 or it<250*args.n_steps) else 1
                    w=(step+1)**2
                    total_w+=w
                
                    #loss_2+= w* (pointwise_l1_error*((1-hole)*new_mask).view(B*N,H,W)).sum()/s
                    loss_2+= w* (pointwise_l1_error*((1-hole)).view(B*N,H,W)).sum()/(1-hole).sum()
                    
                    loss_tv+= w* (torch.abs(reconstruction_flows[:,:,:-1,:-1]-reconstruction_flows[:,:,1:,:-1])+torch.abs(reconstruction_flows[:,:,:-1,:-1]-reconstruction_flows[:,:,:-1,1:])).mean()
                    cumsum=0.
                    for j in range(1,N-1):
                        cumsum+=torch.mean(torch.min(fb(reconstruction_flows.view(B,N,4,H,W)[:,j,:2],reconstruction_flows.view(B,N,4,H,W)[:,j+1,2:],test=True),fb(reconstruction_flows.view(B,N,4,H,W)[:,j,2:],reconstruction_flows.view(B,N,4,H,W)[:,j-1,:2],test=True)))
                    minfbbf+=w* cumsum/(N-2)
                    
                    
                    final_rec=(pointwise_l1_error*((1-hole)).view(B*N,H,W)).sum()/(1-hole).sum()
                    
                    
                elif step==args.n_steps-1:
                    total_w=1
                    loss_2=  (pointwise_l1_error*((1-hole)*new_mask).view(B*N,H,W)).sum()/((1-hole)*new_mask).sum()
                #loss_3+=w*torch.clamp((1/3-new_mask),min=0)[hole!=1].mean()
                
                
                
                     #we add the weight  (1-np.exp(-it/(1000*(step+1)))) that makes loss2 be 0 for the first batch iteration, then around 1000, we account for the first step of the iterative scheme; and every 1000, another step is smoothly taken into account.
                     #justification: in the beginning, we want the translation between fows and features to converge before using the predicted flows to update the features. Alos, since the update is recurrent, we want to train well the first steps of it before adding a new one
                
                
                #F=F*(new_mask<=mask_current)+new_F*(new_mask>mask_current) no need now
                

                if i % args.PRINT_EVERY == 0:#TESTING CODE
                    for frame in range(args.n_frames):
                        if not os.path.exists(log_dir+"/test1"):
                            os.makedirs(log_dir+"/test1")
                        if not os.path.exists(log_dir+"/mask_frame_"+str(frame)):
                            os.makedirs(log_dir+"/mask_frame_"+str(frame))
                        if not os.path.exists(log_dir+"/test2"):
                            os.makedirs(log_dir+"/test2")
                        if not os.path.exists(log_dir+"/forward_frame_"+str(frame)):
                            os.makedirs(log_dir+"/forward_frame_"+str(frame))
                        if not os.path.exists(log_dir+"/backward_frame_"+str(frame)):
                            os.makedirs(log_dir+"/backward_frame_"+str(frame))
                        with torch.no_grad():
                            out=(F2flow_dict[0](new_F[:,frame])*(new_mask[:,frame]>0)).cpu().numpy()
                        display=torch.ones(new_mask.shape[2:]).cuda()
                        #print(new_mask[0,0].shape,display.shape)
                        display[0,0,0]=0
                        for j in range(B):#batch
                            iio.write(log_dir+'/test1/{:02d}_{:02d}.flo'.format(j,step),t1[N*j].detach().cpu().numpy().transpose(1,2,0))
                            iio.write(log_dir+'/test2/{:02d}_{:02d}.flo'.format(j,step),t2[N*j].detach().cpu().numpy().transpose(1,2,0))
                            
                            iio.write(log_dir+'/mask_frame_'+str(frame)+'/{:02d}_{:02d}.png'.format(j,step),(display*new_mask[j,frame]).detach().cpu().numpy().transpose(1,2,0)*255.)
                            iio.write(log_dir+'/forward_frame_'+str(frame)+'/{:02d}_{:02d}.flo'.format(j,step),out[j,:2].transpose(1,2,0))
                            iio.write(log_dir+'/backward_frame_'+str(frame)+'/{:02d}_{:02d}.flo'.format(j,step),out[j,2:].transpose(1,2,0))
                            
                            if step==args.n_steps-1:
                                iio.write(log_dir+'/test1/{:02d}_{:02d}.flo'.format(j,args.n_steps),t1[N*j].detach().cpu().numpy().transpose(1,2,0))
                                iio.write(log_dir+'/test2/{:02d}_{:02d}.flo'.format(j,args.n_steps),t2[N*j].detach().cpu().numpy().transpose(1,2,0))
                                
                                iio.write(log_dir+'/mask_frame_'+str(frame)+'/{:02d}_{:02d}.png'.format(j,args.n_steps),mask_initial[j,frame].detach().cpu().numpy().transpose(1,2,0)*255.)
                                iio.write(log_dir+'/forward_frame_'+str(frame)+'/{:02d}_{:02d}_gt.flo'.format(j,args.n_steps),gt_all_flow[j,frame].cpu().numpy().transpose(1,2,0))
                                iio.write(log_dir+'/backward_frame_'+str(frame)+'/{:02d}_{:02d}_gt.flo'.format(j,args.n_steps),gt_all_rflow[j,frame].cpu().numpy().transpose(1,2,0))
                                
                                
                           
                
                F=new_F*1.
                mask_current=torch.clamp(new_mask,0,1)*1.#mask update befor next step

            loss_1= pointwise_l1_error[(hole==1).view(B*N,H,W)].mean()
            #loss_2= (pointwise_l1_error*((1-hole)*new_mask.detach()).view(B*N,H,W)).sum()/((1-hole)*new_mask).sum()
            loss_3= torch.clamp(.5-new_mask[hole==0].mean(),min=0)
            loss_2=loss_2/total_w
            loss_tv=loss_tv/total_w
            minfbbf=minfbbf/total_w
            # loss_1=loss_1/args.n_steps
            #loss_3=loss_3/total_w
            optimizer.zero_grad()

            (1.*loss_1+(1-np.exp(-it/1000))*(1.*loss_2+1.*loss_tv+1.*minfbbf)+0.*loss_3).backward()#weighting of the loss

            optimizer.step()
            
            loss['flow_l1_reconstruction']=loss_1.detach()
            loss['min(fb,bf)']=minfbbf.detach()
            loss['masked_l1_prediction_error']=loss_2.detach()
            loss['final reconstruction']=final_rec.detach()
            loss['mask fill loss']=loss_3.detach()
            loss['tv loss']=loss_tv.detach()
            #print(loss_1,loss_2)
            loss['p']=10*net.pconv1.p.detach()
            loss['p2']=10*net.pconv2.p.detach()
            loss['a1']=10*net.pconv1.a.detach()
            loss['b1']=10*net.pconv1.b.detach()
            if 'initial_mask' in args.model_name:
                #loss['t']=100*net.t.detach()
                #loss['c']=100*net.c.detach()
                loss['initial_mask mean']=mask_initial[hole==0].mean().detach().cpu()
            
            write_loss_dict(loss, writer, it)
            if (it+1) % args.MODEL_SAVE_STEP == 0:
                save_ckpt(os.path.join(model_save_dir, 'update_%d.pth' % it),
                        [('update',net)], [('optimizer', optimizer)], it)
                save_ckpt(os.path.join(model_save_dir, 'flow_2F_%d.pth' % it),
                        [('flow2F',flow2F)], [('optimizer', optimizer)], it)
                
                save_ckpt(os.path.join(model_save_dir, 'F2flow_%d.pth' % it),
                        [('F2flow',F2flow_dict[0])], [('optimizer', optimizer)], it)

                print('Model has been saved at %d Iters' % it)
    writer.close()

    
    

if __name__ == '__main__':
    main()

    
    
    
    
    
    
    
    
    
    
    
    
    
