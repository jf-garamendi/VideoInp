import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
import argparse
import yaml

import matplotlib.pyplot as plt
import torch
import cvbase as cvb
import numpy as np
import cv2
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from torch.utils import data
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from torch.utils.data import DataLoader

from models import *
from utils.runner_func import *
from utils.full_data_new import *
from utils.losses import *
from utils.io import *
import iio
parser = argparse.ArgumentParser()

# training options
parser.add_argument('--save_dir', type=str, default='./snapshots/default')
parser.add_argument('--log_dir', type=str, default='./logs/default')
parser.add_argument('--model_name', type=str, default=None)

parser.add_argument('--max_iter', type=int, default=100000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_threads', type=int, default=8)
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
parser.add_argument('--MASK_MODE', type=str, default=None)
parser.add_argument('--PRETRAINED', action='store_true')
parser.add_argument('--PRETRAINED_MODEL', type=str, default=None)
parser.add_argument('--RESNET_PRETRAIN_MODEL', type=str,
                    default='pretrained_models/resnet50-19c8e357.pth')
parser.add_argument('--TRAIN_LIST', type=str, default='./VOS_test')
parser.add_argument('--TEST_LIST', type=str, default='./full_list_test')
parser.add_argument('--EVAL_LIST', type=str, default=None)
parser.add_argument('--MASK_ROOT', type=str, default=None)
parser.add_argument('--DATA_ROOT', type=str, default=None,
                    help='Set the path to flow dataset')
parser.add_argument('--INITIAL_HOLE', action='store_true')
parser.add_argument('--TRAIN_LIST_MASK', type=str, default=None)

parser.add_argument('--PRINT_EVERY', type=int, default=5)
parser.add_argument('--MODEL_SAVE_STEP', type=int, default=5000)
parser.add_argument('--NUM_ITERS_DECAY', type=int, default=10000)
parser.add_argument('--CPU', action='store_true')

parser.add_argument('--MASK_HEIGHT', type=int, default=120)
parser.add_argument('--MASK_WIDTH', type=int, default=212)
parser.add_argument('--VERTICAL_MARGIN', type=int, default=10)
parser.add_argument('--HORIZONTAL_MARGIN', type=int, default=10)
parser.add_argument('--MAX_DELTA_HEIGHT', type=int, default=60)
parser.add_argument('--MAX_DELTA_WIDTH', type=int, default=106)
parser.add_argument('--rflow', action='store_true')
parser.add_argument('--two_masks', action='store_true')
parser.add_argument('--n_frames', type=int, default=6)
parser.add_argument('--re',type=str, default="")
parser.add_argument('--image', action='store_true')
parser.add_argument('--edge', type=str, default=None)

parser.add_argument('--seg', type=str, default=None)
parser.add_argument('--force', type=str, default=None)
parser.add_argument('--avoid', type=str, default=None)
parser.add_argument('--bbox', action='store_true')
parser.add_argument('--bias_edge', action='store_true')
parser.add_argument('--n_steps', type=int, default=10)

args = parser.parse_args()
import re


def main():
    
    

    image_size = [args.IMAGE_SHAPE[0], args.IMAGE_SHAPE[1]]
    torch.manual_seed(7777777)
    if not args.CPU:
        torch.cuda.manual_seed(7777777)
        

    
    log_dir="./logs_iterative"
    model_save_dir="./snapshots"
    
    train_dataset = dataset(args,test=True)
    train_loader = DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            num_workers=args.n_threads)
    train_iterator = iter(train_loader)

   
    for i, (flow_masked,rflow_masked, mask, gt_all_flow, gt_all_rflow, edge, img, bbox) in enumerate(train_loader):
        hole=1-mask[:,:,:1].cuda()
        mask_initial=1-1.*mask[:,:,:1].cuda()
        mask_current=1.*mask_initial
        gt_all_rflow=gt_all_rflow.cuda()
        gt_all_flow=gt_all_flow.cuda()
        B,N,C,H,W=flow_masked.shape
        
        flow=flow_masked.cuda()
        rflow=rflow_masked.cuda()
        flows=torch.cat((flow,rflow),dim=2)
        gt_flows=torch.cat((gt_all_flow,gt_all_rflow),dim=2)

        xx,yy=torch.meshgrid(torch.arange(H),torch.arange(W))
        ind=torch.stack((yy,xx),dim=-1)
        ind=ind.repeat(B,1,1,1).cuda()
        with torch.no_grad():
            for model in os.listdir(log_dir):
                if re.match(args.re,model) and not re.match(r'^(?!.*fill).*',model):
    
                    print(model)
                    #dir=os.path.join('./tests',model)
                    n=0
                    for s in os.listdir(os.path.join(model_save_dir,model,"ckpt")):
                        if 'update' in s:
                            n=max(n,int(s.strip("update_").strip(".pth")))
                    print(n)
                    if n==0:
                        print('no save')
                    else:
                        if 'pow' in model:
                            args.update='pow'
                        elif 'pol' in model:
                            args.update='pol'
                        
                        if '2' in model:
                            net=Res_Update2(update=args.update)
                        elif '3' in model:
                            net=Res_Update3(update=args.update)
                        elif '4' in model:
                            net=Res_Update4(update=args.update)
                        
                        flow2F=Flow2features().cuda().eval()
                        F2flow=Features2flow().cuda().eval()
                        net.cuda().eval()
                        
                        resume_iter = load_ckpt(os.path.join(model_save_dir,model,"ckpt","flow_2F_"+str(n)+".pth"),[('flow2F', flow2F)],strict=True)
                        resume_iter = load_ckpt(os.path.join(model_save_dir,model,"ckpt","F2flow_"+str(n)+".pth"),[('F2flow', F2flow)],strict=True)
                        resume_iter = load_ckpt(os.path.join(model_save_dir,model,"ckpt","update_"+str(n)+".pth"),[('update', net)],strict=True)
                        print('Model Resume from', resume_iter, 'iter')
                        
                    
                        start_iter = 0 
                    
                        F=flow2F(flows.view(B*N,2*C,H,W)).view(B,N,32,H,W)
                        
        
                        
                        for step in range(args.n_steps):
                            
                            new_mask=mask_current*0.
                            new_F=F*0.
                            for frame in range(N):
                                
                                if frame+1<N:
                                    
                                    grid_f=F2flow(F[:,frame])[:,:2].permute((0,2,3,1))+ind #compute forward flow from features
                                    grid_f=(2*grid_f/torch.tensor([W,H]).cuda().view(1,1,1,2))-1
                                    F_f=torch.nn.functional.grid_sample(F[:,frame+1],grid_f,mode='bilinear',padding_mode='border',align_corners=False)
                                    mask_f=torch.clamp(torch.nn.functional.grid_sample(mask_current[:,frame+1],grid_f,mode='bilinear',padding_mode='border',align_corners=False),0,1)
                                    #compute interpolation of the features and masks at frame t+1 (if t+1 is a video frame) following the predicted forward flow
                                else:
                                    F_f=0.*F[:,frame]
                                    mask_f=0.*mask_current[:,frame].cuda()
                            
                                if frame-1>=0:
                                    grid_b=F2flow(F[:,frame])[:,2:].permute((0,2,3,1))+ind #compute backward flow from features
                                    grid_b=(2*grid_b/torch.tensor([W,H]).cuda().view(1,1,1,2))-1
                                    F_b=torch.nn.functional.grid_sample(F[:,frame-1],grid_b,mode='bilinear',padding_mode='border',align_corners=False)
                                    mask_b=torch.clamp(torch.nn.functional.grid_sample(mask_current[:,frame-1],grid_b,mode='bilinear',padding_mode='border',align_corners=False),0,1)
                                else:
                                    F_b=0.*F[:,frame]
                                    mask_b=0.*mask_current[:,frame].cuda()
                                
                                x=torch.cat((F_b,F[:,frame],F_f),dim=1)#input of the update network is the concatenation of the obtained features from this frame and the neighboring ones
                                mask_in=torch.cat(((mask_b).repeat(1,F.shape[2],1,1),mask_current[:,frame].repeat(1,F.shape[2],1,1),(mask_f).repeat(1,F.shape[2],1,1)),dim=1)#same goes for the input mask
                                new_F[:,frame],new_mask[:,frame]=net(x,mask_in)#Update
                                new_mask[:,frame][hole[:,frame]==1]=1.#force the initially confident pixels to stay confident, because a decay can be observed depending on the update rule of the partial convolution
                            
                        
                    
                            dir=os.path.join('./tests/new_comp_iter/'+str(model))
            
                            if not os.path.exists(dir):
                                os.makedirs(dir)
                            for frame in range(args.n_frames):
                                
                                if not os.path.exists(dir+"/mask_frame_"+str(frame)):
                                    os.makedirs(dir+"/mask_frame_"+str(frame))
                                
                                if not os.path.exists(dir+"/forward_frame_"+str(frame)):
                                    os.makedirs(dir+"/forward_frame_"+str(frame))
                                if not os.path.exists(dir+"/backward_frame_"+str(frame)):
                                    os.makedirs(dir+"/backward_frame_"+str(frame))
                                with torch.no_grad():
                                    out=(F2flow(new_F[:,frame])*(new_mask[:,frame]>0)).cpu().numpy()
                                display=torch.ones(new_mask.shape[2:])
                                display[0,0]=0
                                for j in range(B):#batch
        
                                    
                                    iio.write(dir+'/mask_frame_'+str(frame)+'/{:02d}_{:02d}.png'.format(j,step),display*new_mask[j,frame].detach().cpu().numpy().transpose(1,2,0)*255.)
                                    iio.write(dir+'/forward_frame_'+str(frame)+'/{:02d}_{:02d}.flo'.format(j,step),out[j,:2].transpose(1,2,0))
                                    iio.write(dir+'/backward_frame_'+str(frame)+'/{:02d}_{:02d}.flo'.format(j,step),out[j,2:].transpose(1,2,0))
                                    
                                    if step==args.n_steps-1:
                                    
                                        iio.write(dir+'/mask_frame_'+str(frame)+'/{:02d}_{:02d}.png'.format(j,args.n_steps),mask_initial[j,frame].detach().cpu().numpy().transpose(1,2,0)*255.)
                                        iio.write(dir+'/forward_frame_'+str(frame)+'/{:02d}_{:02d}_gt.flo'.format(j,args.n_steps),gt_all_flow[j,frame].cpu().numpy().transpose(1,2,0))
                                        iio.write(dir+'/backward_frame_'+str(frame)+'/{:02d}_{:02d}_gt.flo'.format(j,args.n_steps),gt_all_rflow[j,frame].cpu().numpy().transpose(1,2,0))
                                        
                                        
                                
                            
                            F=new_F*1.
                            mask_current=new_mask*1.#mask update befor next step
                        
            # 
            # dir=os.path.join('./tests/new_comp_iter/gt')
    
        #   if not os.path.exists(dir):
            #     os.makedirs(dir)
            # for j in range(flows[0].shape[0]):
            #     for k in range(args.n_frames):#layer
    
        #           iio.write(dir+'/forward_{:02d}_{:02d}.flo'.format(j,k),fgt[j,k].transpose(1,2,0))
            #         iio.write(dir+'/backward_{:02d}_{:02d}.flo'.format(j,k),bgt[j,k].transpose(1,2,0))
            input('next batch')

    

        
if __name__ == '__main__':
    main()
        
        
        
        
        
        
        
        
        