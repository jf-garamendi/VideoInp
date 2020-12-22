#1 'test_l1_Wpatch_GAN' loss l1 patchgan <- input flows, masks [2,3]
#2 'test_l1_fb' loss l1 gan <- input flows, masks [2,3]
#3




f f fp fp f f
b b bp bp b b
##
with open('/home/pierrick/Documents/GitHub/data/Youtube_VOS_2018/train_all_frames/JPEGImages/0a2f2bd294/Flow/minmax.txt','r') as f :
    for line in f:
        line=line.rstrip()
        print(line)

##
import numpy as np
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
parser.add_argument('--EVAL_LIST', type=str, default='/home/pierrick/Documents/Projects/UNet/full_list_test')
parser.add_argument('--TEST_LIST', type=str, default='/home/pierrick/Documents/Projects/UNet/full_list_test')
parser.add_argument('--MASK_ROOT', type=str, default=None)
parser.add_argument('--DATA_ROOT', type=str, default=None,
                    help='Set the path to flow dataset')
parser.add_argument('--INITIAL_HOLE', action='store_true')

parser.add_argument('--PRINT_EVERY', type=int, default=5)
parser.add_argument('--MODEL_SAVE_STEP', type=int, default=10000)
parser.add_argument('--NUM_ITERS_DECAY', type=int, default=1000)

parser.add_argument('--MASK_HEIGHT', type=int, default=120)
parser.add_argument('--MASK_WIDTH', type=int, default=212)
parser.add_argument('--VERTICAL_MARGIN', type=int, default=10)
parser.add_argument('--HORIZONTAL_MARGIN', type=int, default=10)
parser.add_argument('--MAX_DELTA_HEIGHT', type=int, default=100)#60
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
parser.add_argument('--n_frames', type=int, default=5)
parser.add_argument('--n_critic', type=int, default=5)
parser.add_argument('--two_masks', action='store_true')
parser.add_argument('--bias_edge', action='store_true')
parser.add_argument('--sandwich', action='store_true')
parser.add_argument('--bbox', action='store_true')


args = parser.parse_args()
args.FIX_MASK=True
args.MASK_HEIGHT=50 
args.MASK_WIDTH=50 
args.MAX_DELTA_HEIGHT=10
args.MAX_DELTA_WIDTH=10
args.MASK_MODE='bbox'
args.two_masks=True
args.rflow=True
args.edge='canny'
args.image=True
args.bias_edge=True
args.n_threads=0
val_dataset = dataset(args,test=True)
val_loader = DataLoader(val_dataset,
                            batch_size=16,
                            shuffle=True,
                            drop_last=True,
                            num_workers=args.n_threads)
import time
tic=time.time()
a,b,c,d,e,f,g,h=next(iter(val_loader))
print(time.time()-tic)

##       
log_dir='/home/pierrick/Desktop/full_test'
if not os.path.exists(log_dir+"/forward"):
    os.makedirs(log_dir+"/forward")
if not os.path.exists(log_dir+"/minfbbf"):
    os.makedirs(log_dir+"/minfbbf")
if not os.path.exists(log_dir+"/forward_hole"):
    os.makedirs(log_dir+"/forward_hole")
if not os.path.exists(log_dir+"/backward"):
    os.makedirs(log_dir+"/backward")          
if not os.path.exists(log_dir+"/image"):
    os.makedirs(log_dir+"/image")     
if not os.path.exists(log_dir+"/fb"):
    os.makedirs(log_dir+"/fb")     
if not os.path.exists(log_dir+"/edge"):
    os.makedirs(log_dir+"/edge")
if not os.path.exists(log_dir+"/grad_norm"):
    os.makedirs(log_dir+"/grad_norm")        
if not os.path.exists(log_dir+"/edge_loss"):
    os.makedirs(log_dir+"/edge_loss")  
if not os.path.exists(log_dir+"/Pablo_loss"):
    os.makedirs(log_dir+"/Pablo_loss") 
if not os.path.exists(log_dir+"/Pablo_loss_dt"):
    os.makedirs(log_dir+"/Pablo_loss_dt") 
if not os.path.exists(log_dir+"/Pablo_loss_2frames"):
    os.makedirs(log_dir+"/Pablo_loss_2frames") 
if not os.path.exists(log_dir+"/consistent_edge"):
    os.makedirs(log_dir+"/consistent_edge")
d=d.cpu()
grad_norm=torch.sum((d[:,:,:,1:,:-1]-d[:,:,:,:-1,:-1])**2 + (d[:,:,:,:-1,1:]-d[:,:,:,:-1,:-1])**2,dim=2,keepdim=True).sqrt()
N=.5

l=(1-(f[:,:,:,:-1,:-1]/2+.5))*(1-torch.exp(-grad_norm/N))+(f[:,:,:,:-1,:-1]/2+.5)*torch.exp(-grad_norm/N)



for i in range(8):
    for j in range(5):
        iio.write(log_dir+'/forward_hole/{:02d}_{:02d}.flo'.format(i,j),np.array(a[i,j]).transpose(1,2,0))
        iio.write(log_dir+'/forward/{:02d}_{:02d}.flo'.format(i,j),np.array(d[i,j]).transpose(1,2,0))
        iio.write(log_dir+'/backward/{:02d}_{:02d}.flo'.format(i,j),np.array(e[i,j]).transpose(1,2,0))
        iio.write(log_dir+'/image/{:02d}_{:02d}.png'.format(i,j),(np.array(g[i,j]).transpose(1,2,0)+1)*127.5)
        iio.write(log_dir+'/edge/{:02d}_{:02d}.png'.format(i,j),(np.array(f[i,j]).transpose(1,2,0)+1)*127.5)
        iio.write(log_dir+'/grad_norm/{:02d}_{:02d}.png'.format(i,j),np.array(grad_norm[i,j]).transpose(1,2,0)*20)
        iio.write(log_dir+'/edge_loss/{:02d}_{:02d}.png'.format(i,j),np.array(l[i,j]).transpose(1,2,0)*256)
d=d.cuda()
for j in range(4):
    dist=.5*(fb(d[:,j].cuda(),e[:,j+1].cuda(),test=True)+fb(e[:,j+1].cuda(),d[:,j].cuda(),test=True))
    
    xx,yy=torch.meshgrid(torch.arange(d.shape[-2]),torch.arange(d.shape[-1]))
    ind=torch.stack((yy,xx),dim=-1).cuda()
    ind=ind.repeat(d.shape[0],1,1,1).cuda()
    grid=d[:,j].permute(0,2,3,1)+ind
    grid=(2*grid/torch.tensor([d.shape[-1]*1.,d.shape[-2]*1.]).cuda().view(1,1,1,2))-1
    interp=torch.nn.functional.grid_sample(d[:,j+1],grid,mode='bilinear',padding_mode='border',align_corners=False)
    
    
    diff=interp-d[:,j]
    spacial_jac_frob=((diff[:,0,1:,:-1]-diff[:,0,:-1,:-1])**2 + (diff[:,0,:-1,1:]-diff[:,0,:-1,:-1])**2+(diff[:,1,1:,:-1]-diff[:,1,:-1,:-1])**2 + (diff[:,1,:-1,1:]-diff[:,1,:-1,:-1])**2).sqrt()
    if j<3:
        min=torch.min(fb(d[:,j+1].cuda(),e[:,j+2].cuda(),test=True),fb(e[:,j+1].cuda(),d[:,j].cuda(),test=True))
        grid2=ind+d[:,j].permute(0,2,3,1)+interp.permute(0,2,3,1)
        interp2=torch.nn.functional.grid_sample(d[:,j+2],grid,mode='bilinear',padding_mode='border',align_corners=False)
        diff2=-interp2+2*interp-d[:,j]
        spacial_jac_frob2=((diff2[:,0,1:,:-1]-diff2[:,0,:-1,:-1])**2 + (diff2[:,0,:-1,1:]-diff2[:,0,:-1,:-1])**2+(diff2[:,1,1:,:-1]-diff2[:,1,:-1,:-1])**2 + (diff2[:,1,:-1,1:]-diff2[:,1,:-1,:-1])**2).sqrt()
    for i in range(8):
        iio.write(log_dir+'/Pablo_loss_dt/{:02d}_{:02d}.flo'.format(i,j),np.array(diff[i].cpu()).transpose(1,2,0))
        iio.write(log_dir+'/fb/{:02d}_{:02d}.png'.format(i,j),np.array(dist[i].cpu())*20)
        
        
        
        iio.write(log_dir+'/Pablo_loss/{:02d}_{:02d}.png'.format(i,j),np.array(spacial_jac_frob[i].cpu())*20)
        iio.write(log_dir+'/consistent_edge/{:02d}_{:02d}.png'.format(i,j),(1-np.exp(0.01*np.array(spacial_jac_frob[i].cpu())/np.array(spacial_jac_frob[i].cpu().mean())))*np.array((1-torch.exp(0.1*grad_norm[i,j]/grad_norm[i,j].mean())))[0])
        
        if j<3:
            iio.write(log_dir+'/minfbbf/{:02d}_{:02d}.png'.format(i,j),np.array(min[i].cpu())*20)
            iio.write(log_dir+'/Pablo_loss_2frames/{:02d}_{:02d}.png'.format(i,j),np.array(spacial_jac_frob2[i].cpu())*20)
        else:
            iio.write(log_dir+'/minfbbf/{:02d}_{:02d}.png'.format(i,j),np.array(min[i].cpu())*0)
            iio.write(log_dir+'/Pablo_loss_2frames/{:02d}_{:02d}.png'.format(i,3),np.array(spacial_jac_frob2[i].cpu())*0.)

for i in range(8):
    iio.write(log_dir+'/consistent_edge/{:02d}_{:02d}.png'.format(i,4),(1-np.exp(0.1*np.array(spacial_jac_frob[i].cpu())/np.array(spacial_jac_frob[i].cpu().mean())))*np.array((1-torch.exp(0.1*grad_norm[i,j]/grad_norm[i,j].mean())))[0]*0.)
    iio.write(log_dir+'/Pablo_loss_dt/{:02d}_{:02d}.flo'.format(i,4),np.array(diff[i].cpu()).transpose(1,2,0)*0.)
    iio.write(log_dir+'/fb/{:02d}_{:02d}.png'.format(i,4),np.array(dist[i].cpu())*0.)
    iio.write(log_dir+'/Pablo_loss/{:02d}_{:02d}.png'.format(i,4),np.array(spacial_jac_frob[i].cpu())*0.)
    iio.write(log_dir+'/Pablo_loss_2frames/{:02d}_{:02d}.png'.format(i,4),np.array(spacial_jac_frob[i].cpu())*0.)
    iio.write(log_dir+'/minfbbf/{:02d}_{:02d}.png'.format(i,4),np.array(min[i].cpu())*0)
##
for j in range(2):
    for i in range(10):
        if i ==5:
            break
        print(j,i)
        
##

for i, (flow_masked,rflow_masked, mask, gt_all_flow, gt_all_rflow, edge, img, bbox) in enumerate(train_loader):
    for model in os.listdir(log_dir):
        if re.match(args.re,model):

            print(model)
            #dir=os.path.join('./tests',model)
            n=0
            for s in os.listdir(os.path.join(model_save_dir,model,"ckpt")):
                n=max(n,int(s.strip("DFI_").strip(".pth")))
            print(n)
                    
            in_c=30
            print(model)
            if 'res' in model:
                net=Res(2,2,10)
            elif 'UNet' in model:
                net=UNet(2,2)
       
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
            input_x =gt_all_flow[:,0].cuda()
            if args.seg=='b':
                gt_warp= gt_all_rflow[:,1].cuda()
                
            with torch.no_grad():
                flows=net(input_x)[:8]

            
        
            dir=os.path.join('./tests/new_comp_warp/'+str(model))

            if not os.path.exists(dir):
                os.makedirs(dir)

            
            fgt=gt_all_flow[:,:,:,:].detach().cpu().numpy()
            bgt=gt_all_rflow[:,:,:,:].detach().cpu().numpy() 
        
            for j in range(flows[0].shape[0]):#batch
                
                
                iio.write(dir+'/forward/{:02d}.flo'.format(j),input_x[j].detach().cpu().numpy().transpose(1,2,0))
                iio.write(dir+'/backward/{:02d}.flo'.format(j),gt_warp[j].detach().cpu().numpy().transpose(1,2,0))
                iio.write(dir+'/output/{:02d}.flo'.format(j),flows[0][j].detach().cpu().numpy().transpose(1,2,0))



        
        
        
        
        
        
        