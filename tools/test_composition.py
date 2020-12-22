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
from tensorboardX import SummaryWriter
from torch.utils import data
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from torch.utils.data import DataLoader

from models import *
from utils.runner_func import *
from utils.data import *
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
parser.add_argument('--MASK_MODE', type=str, default=None)
parser.add_argument('--PRETRAINED', action='store_true')
parser.add_argument('--PRETRAINED_MODEL', type=str, default=None)
parser.add_argument('--RESNET_PRETRAIN_MODEL', type=str,
                    default='pretrained_models/resnet50-19c8e357.pth')
parser.add_argument('--TRAIN_LIST', type=str, default='./VOS_test')
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

parser.add_argument('--re',type=str, default="")

parser.add_argument('--edge', type=str, default=None)

parser.add_argument('--seg', type=str, default=None)
parser.add_argument('--force', type=str, default=None)
parser.add_argument('--avoid', type=str, default=None)


args = parser.parse_args()
import re


def main():
    
    

    image_size = [args.IMAGE_SHAPE[0], args.IMAGE_SHAPE[1]]
    torch.manual_seed(7777777)
    if not args.CPU:
        torch.cuda.manual_seed(7777777)
        

    
    log_dir="/home/pierrick/Documents/Projects/UNet/tests/edge_then_flow"
    model_save_dir="./snapshots"

    args.SEG_INPUT=False
    args.BOUNDARY=False


    E=Edge_generator(2,1).cuda().eval()
    resume_iter = load_ckpt(os.path.join("~/Documents/Projects/UNet/snapshots/Edge_WGAN.1/ckpt","DFI_29999.pth"),[('model', E)],strict=True)
    G=UNet(4,2).cuda.eval()
    resume_iter = load_ckpt(os.path.join("~/Documents/Projects/UNet/snapshots/UNet_canny_l1_WpatchGAN.1/ckpt","DFI_79999.pth"),[('model', G)],strict=True)
    
    args.edge='canny'
    train_dataset = dataset(args)
    train_loader = DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=True,
                            num_workers=args.n_threads)
    train_iterator = iter(train_loader)

    start_iter = 0 

    with torch.no_grad():
        for i in tqdm(range(start_iter, min(len(train_loader),50))):
            flow_mask_cat, flow_masked, gt_flow, mask,_ = next(train_iterator)
            
            input_x = flow_mask_cat.cuda()
            gt_flow = gt_flow.cuda()
            mask = mask.cuda()
            flow_masked = flow_masked.cuda()
        
            input_x_masked_edge=input_x[:,2:]*1.
            input_x_masked_edge[:,1:] = input_x[:,3:,:,:] * (1. - mask[:,0:1,:,:])
            edges = E(input_x_masked_edge)[0]
            
            input_G=input_x*1.
            input_G[:,3:]=edges
            flows = G(input_G)
    
            fake = (flows[-1] * mask[:,:,:,:] + flow_masked[:,:,:,:] * (1. - mask[:,:,:,:])).detach().cpu().numpy()
        

            dir="/home/pierrick/Documents/Projects/UNet/tests/edge_then_flow"
            dir_in="/home/pierrick/Documents/Projects/UNet/tests/edge_then_flow"
            if not os.path.exists(dir_in):
                os.makedirs(dir_in)
            if not os.path.exists(dir):
                os.makedirs(dir)
                
                
            
            if not os.path.exists(dir+"/fake"):
                os.makedirs(dir+"/fake")
            if not os.path.exists(dir_in+"/input"):
                os.makedirs(dir_in+"/input")
            if not os.path.exists(dir+"/output"):
                os.makedirs(dir+"/output")
            if not os.path.exists(dir_in+"/gt"):
                os.makedirs(dir_in+"/gt")
            if not os.path.exists(dir_in+"/canny"):
                os.makedirs(dir_in+"/canny")
            
            if not os.path.exists(dir_in+"/seg_gt"):
                os.makedirs(dir_in+"/seg_gt")
            
        
            input=flow_masked[:,:2,:,:].detach().cpu().numpy()
            gt=gt_flow[:,:,:,:].detach().cpu().numpy()
            seg=input_x[:,3:,:,:].detach().cpu().numpy()
            
            output=flows[-1].detach().cpu().numpy()
        
            for j in range(args.batch_size):
                iio.write(dir+'/fake/{:03d}.flo'.format(j),fake[j].transpose(1,2,0))
                iio.write(dir_in+'/gt/{:03d}.flo'.format(j),gt[j].transpose(1,2,0))
                iio.write(dir_in+'/input/{:03d}.flo'.format(j),input[j].transpose(1,2,0))
                iio.write(dir+'/output/{:03d}.flo'.format(j),output[j].transpose(1,2,0))
                if args.edge=='grad':
                    iio.write(dir_in+'/seg/{:03d}.png'.format(j),seg[j].transpose(1,2,0)*256+256)
                elif args.edge=='hed':
                    iio.write(dir_in+'/hed/{:03d}.png'.format(j),seg[j].transpose(1,2,0)*256+256)
                elif args.seg=='gt':
                    iio.write(dir_in+'/seg_gt/{:03d}.png'.format(j),seg[j].transpose(1,2,0)*256+256)
                elif args.edge=='canny':
                    iio.write(dir_in+'/canny/{:03d}.png'.format(j),seg[j].transpose(1,2,0)*256+256)


        

        
if __name__ == '__main__':
    main()
        
        
        
        
        
        
        
        
        