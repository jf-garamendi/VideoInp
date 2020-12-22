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
parser.add_argument('--MAX_DELTA_HEIGHT', type=int, default=100)
parser.add_argument('--MAX_DELTA_WIDTH', type=int, default=180)
parser.add_argument('--rflow', action='store_true')
parser.add_argument('--two_masks', action='store_true')
parser.add_argument('--n_frames', type=int, default=6)
parser.add_argument('--re',type=str, default="")
parser.add_argument('--image', action='store_true')
parser.add_argument('--edge', type=str, default=None)

parser.add_argument('--seg', type=str, default=None)
parser.add_argument('--force', type=str, default=None)
parser.add_argument('--avoid', type=str, default=None)

parser.add_argument('--bias_edge', action='store_true')
parser.add_argument('--bbox', action='store_true')
parser.add_argument('--n_steps', type=int, default=5)
parser.add_argument('--update', type=str, default='pow')
parser.add_argument('--schedule', action='store_true')
parser.add_argument('--sandwich', action='store_true')

args = parser.parse_args()
import re


def main():
    
    

    image_size = [args.IMAGE_SHAPE[0], args.IMAGE_SHAPE[1]]
    torch.manual_seed(7777777)
    if not args.CPU:
        torch.cuda.manual_seed(7777777)
        

    
    log_dir="./logs"
    model_save_dir="./snapshots"
    
    train_dataset = dataset(args,test=True)
    train_loader = DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            num_workers=args.n_threads)
    train_iterator = iter(train_loader)

   
    for i, (flow_masked,rflow_masked, mask, gt_all_flow, gt_all_rflow, edge, img, bbox) in enumerate(train_loader):
        for model in os.listdir(log_dir):
            if re.match(args.re,model) and re.match(r'^(?!.*small_mask).*',model):

                print(model)
                #dir=os.path.join('./tests',model)
                n=0
                for s in os.listdir(os.path.join(model_save_dir,model,"ckpt")):
                    n=max(n,int(s.strip("DFI_").strip(".pth")))
                print(n)
                        
                in_c=30
                out_c=8
                net=UNet(in_c,out_c)
    
                resume_iter = load_ckpt(os.path.join(model_save_dir,model,"ckpt","DFI_"+str(n)+".pth"),[('model', net)],strict=True)
                print('Model Resume from', resume_iter, 'iter')
                
                
                
                #TAKE CARE OF EDGE INPUT
                if 'hed' in model:
                    args.edge='hed'
                elif 'grad' in model:
                    args.edge='grad'
                elif 'canny' in model:
                    args.edge='canny'
                else:
                    args.edge=None
                    
                
                start_iter = 0 
                net.cuda().eval()
                
                
            
    
                gt_all_rflow=gt_all_rflow.cuda()
                gt_all_flow=gt_all_flow.cuda()
                B,N,C,H,W=flow_masked.shape
                input_x = torch.cat((flow_masked.view(B,N*C,H,W),rflow_masked.view(B,N*C,H,W),mask[:,:,0]),dim=1).cuda()
                gt_flow = gt_all_flow[:,2:4].view(B,4,H,W).cuda()
                gt_rflow= gt_all_rflow[:,2:4].view(B,4,H,W).cuda()
                mask = mask.cuda()
                fmask=mask[:,2]
                bmask=mask[:,3]
                #
                flow_masked = flow_masked.cuda()
                rflow_masked = rflow_masked.cuda()
                with torch.no_grad():
                    flows=net(input_x)
                fflows=[flows[0][:,:4]]
                bflows=[flows[0][:,4:]]
               

                dir=os.path.join('./tests/new_comp/'+str(model))

                if not os.path.exists(dir):
                    os.makedirs(dir)
                    
                    
                
                # if not os.path.exists(dir+"/forward"):
                #     os.makedirs(dir+"/forward")
                # if not os.path.exists(dir_in+"/backward"):
                #     os.makedirs(dir_in+"/backward")
                # if not os.path.exists(dir+"/output"):
                #     os.makedirs(dir+"/output")
                # if not os.path.exists(dir_in+"/fgt"):
                #     os.makedirs(dir_in+"/fgt")
                # if not os.path.exists(dir_in+"/bgt"):
                #     os.makedirs(dir_in+"/bgt")

                # if not os.path.exists(dir_in+"/canny"):
                #     os.makedirs(dir_in+"/canny")
                
                fgt=gt_all_flow[:,:,:,:].detach().cpu().numpy()
                bgt=gt_all_rflow[:,:,:,:].detach().cpu().numpy() 
            
                for j in range(flows[0].shape[0]):
                    for k in range(args.n_frames):#layer
                        if k==2:
                            ffake = (fflows[0][j,:2] * mask[j,k,:,:] + flow_masked[j,k,:,:,:] * (1. - mask[j,k,:,:])).detach().cpu().numpy()
                            bfake=(bflows[0][j,:2] * mask[j,k,:,:] + rflow_masked[j,k,:,:,:] * (1. - mask[j,k,:,:])).detach().cpu().numpy()
                        elif k==3:
                            ffake = (fflows[0][j,2:] * mask[j,k,:,:] + flow_masked[j,k,:,:,:] * (1. - mask[j,k,:,:])).detach().cpu().numpy()
                            bfake=(bflows[0][j,2:] * mask[j,k,:,:] + rflow_masked[j,k,:,:,:] * (1. - mask[j,k,:,:])).detach().cpu().numpy()
                        else:
                            ffake = flow_masked[j,k,:,:,:].detach().cpu().numpy()
                            bfake=rflow_masked[j,k,:,:,:].detach().cpu().numpy()
                        
                        
                        iio.write(dir+'/forward_{:02d}_{:02d}.flo'.format(j,k),ffake.transpose(1,2,0))
                        iio.write(dir+'/backward_{:02d}_{:02d}.flo'.format(j,k),bfake.transpose(1,2,0))
                        # iio.write(log_dir+'/fgt/{:02d}_{:02d}.flo'.format(j,k),fgt[j,k].transpose(1,2,0))
                        # iio.write(log_dir+'/bgt/{:02d}_{:02d}.flo'.format(j,k),bgt[j,k].transpose(1,2,0))
                        #iio.write(log_dir+'/output{:02d}_{:02d}.flo'.format(j,k),fflows[0][j].detach().cpu().numpy().transpose(1,2,0))
                    
                    
                    
                    
                    # 
                    # 
                
   #                #   input=flow_masked[:,:2,:,:].detach().cpu().numpy()
                    # gt=gt_flow[:,:,:,:].detach().cpu().numpy()
                    # seg=input_x[:,3:,:,:].detach().cpu().numpy()
                    # 
                    # output=flows[-1].detach().cpu().numpy()
                    # 
                    # 
                    # 
                    # for j in range(args.batch_size):
                    #     iio.write(dir+'/fake/{:03d}.flo'.format(j),fake[j].transpose(1,2,0))
                    #     iio.write(dir_in+'/gt/{:03d}.flo'.format(j),gt[j].transpose(1,2,0))
                    #     iio.write(dir_in+'/input/{:03d}.flo'.format(j),input[j].transpose(1,2,0))
                    #     iio.write(dir+'/output/{:03d}.flo'.format(j),output[j].transpose(1,2,0))
                    #     if args.edge=='grad':
                    #         iio.write(dir_in+'/seg/{:03d}.png'.format(j),seg[j].transpose(1,2,0)*256+256)
                    #     elif args.edge=='hed':
                    #         iio.write(dir_in+'/hed/{:03d}.png'.format(j),seg[j].transpose(1,2,0)*256+256)
                    #     elif args.seg=='gt':
                    #         iio.write(dir_in+'/seg_gt/{:03d}.png'.format(j),seg[j].transpose(1,2,0)*256+256)
                    #     elif args.edge=='canny':
                    #         iio.write(dir_in+'/canny/{:03d}.png'.format(j),seg[j].transpose(1,2,0)*256+256)
        dir=os.path.join('./tests/new_comp/gt')

        if not os.path.exists(dir):
            os.makedirs(dir)
        for j in range(flows[0].shape[0]):
            for k in range(args.n_frames):#layer

                iio.write(dir+'/forward_{:02d}_{:02d}.flo'.format(j,k),fgt[j,k].transpose(1,2,0))
                iio.write(dir+'/backward_{:02d}_{:02d}.flo'.format(j,k),bgt[j,k].transpose(1,2,0))
        input('next batch')

    

        
if __name__ == '__main__':
    main()
        
        
        
        
        
        
        
        
        