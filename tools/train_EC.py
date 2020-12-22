import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
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
from utils.runner_func import *
from utils.data import *
from utils.losses import *
from utils.io import *


from skimage.feature import canny



parser = argparse.ArgumentParser()

# training options
parser.add_argument('--net', type=str, default='default')
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
                    default=None)#'pretrained_models/resnet50-19c8e357.pth'
parser.add_argument('--TRAIN_LIST', type=str, default='/home/pierrick/Documents/Projects/UNet/VOS_train')
parser.add_argument('--EVAL_LIST', type=str, default='/home/pierrick/Documents/Projects/UNet/VOS_train')
parser.add_argument('--MASK_ROOT', type=str, default=None)
parser.add_argument('--DATA_ROOT', type=str, default=None,
                    help='Set the path to flow dataset')
parser.add_argument('--INITIAL_HOLE', action='store_true')

parser.add_argument('--PRINT_EVERY', type=int, default=5)
parser.add_argument('--MODEL_SAVE_STEP', type=int, default=5000)
parser.add_argument('--NUM_ITERS_DECAY', type=int, default=1000)

parser.add_argument('--MASK_HEIGHT', type=int, default=120)
parser.add_argument('--MASK_WIDTH', type=int, default=212)
parser.add_argument('--VERTICAL_MARGIN', type=int, default=10)
parser.add_argument('--HORIZONTAL_MARGIN', type=int, default=10)
parser.add_argument('--MAX_DELTA_HEIGHT', type=int, default=80)#60
parser.add_argument('--MAX_DELTA_WIDTH', type=int, default=150)#106

parser.add_argument('--lambda_gan', type=float, default=1/100)
parser.add_argument('--lambda_ld', type=float, default=.1)

parser.add_argument('--seg', type=str, default=None)
parser.add_argument('--edge', type=str, default=None)
parser.add_argument('--loss', type=str, default='epe')
parser.add_argument('--GAN', action='store_true')
parser.add_argument('--patchGAN', action='store_true')
parser.add_argument('--multipatchGAN', action='store_true')
parser.add_argument('--patchGANmask', action='store_true')
parser.add_argument('--WpatchGANmask', action='store_true')
parser.add_argument('--n_critic', type=int, default=5)
args = parser.parse_args()

def main():
    print(args.model_name)
    flat=Flatten()
    if args.max_iter==30000:
        steps=[20000]
    else:
        steps=[40000,60000]
    if args.seg=='gt':
        in_c=6
    else:
        in_c =3
    print(in_c)
    if args.net=='interponet':
        IN=InterpoNet()
    elif args.net=='res':
        IN=Res(in_c,10,10)
    elif args.net=='unet':
        IN=Edge_generator(2,1)
    if args.model_name is not None:
        model_save_dir = './snapshots/'+args.model_name+'/ckpt/'
        sample_dir = './snapshots/'+args.model_name+'/images/'
        log_dir = './logs/'+args.model_name

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir=log_dir)
    torch.cuda.manual_seed(7777777)

    train_dataset = dataset(args)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=args.n_threads)
    val_dataset = dataset(args,val=True)
    val_loader = DataLoader(val_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              drop_last=True,
                              num_workers=args.n_threads)
    train_iterator = iter(train_loader)
        
    
    D=Discriminator(2).cuda().train()
    optimizer_D = optim.Adam(D.parameters(),lr=args.LR,betas=(args.BETA1,args.BETA2))

    IN.cuda().train()
    optimizer = optim.Adam(IN.parameters(),lr=args.LR,betas=(args.BETA1,args.BETA2),weight_decay=args.WEIGHT_DECAY)
    if args.resume:
        if args.PRETRAINED_MODEL is not None:
            resume_iter = load_ckpt(args.PRETRAINED_MODEL,
                                    [('model', IN)],
                                    [('optimizer', optimizer)])
            print('Model Resume from', resume_iter, 'iter')
        else:
            print('Cannot load Pretrained Model')

    loss = {}
    start_iter = 0 if not args.resume else resume_iter

    for i in tqdm(range(start_iter, args.max_iter)):
        # st = time.time()
        # try:
        #     flow_mask_cat, flow_masked, gt_flow, mask, bbox = next(train_iterator)
        # except StopIteration:
        #     batch_iterator = iter(data_loader)
        #     images, targets = next(batch_iterator)
        
        try:
            flow_mask_cat, flow_masked, gt_flow, mask, bbox = next(train_iterator)
        except:
            print('Loader Restart')
            train_iterator = iter(train_loader)
            flow_mask_cat, flow_masked, gt_flow, mask, bbox = next(train_iterator)
            
   
       
        input_x = flow_mask_cat.cuda()
        gt_flow = gt_flow.cuda()
        mask = mask.cuda()
        flow_masked = flow_masked.cuda()

        input_x_masked_edge=input_x[:,2:]*1.
        input_x_masked_edge[:,1:] = input_x[:,3:,:,:] * (1. - mask[:,0:1,:,:])
        flows=IN(input_x_masked_edge)
        edges = flows[0]

        # 
        # loss['epe']= epe(torch.stack(flows,dim=4), torch.stack([gt_flow]*len(flows),dim=4))
        # loss['ld'] = ld(torch.stack(flows,dim=4), torch.stack([gt_flow]*len(flows),dim=4))
        # 
        # loss['L1 mask'] = L1_mask(torch.stack(flows,dim=4), torch.stack([gt_flow]*len(flows),dim=4),torch.stack([mask]*len(flows),dim=4))
        # loss['MSE mask'] = MSE_mask(torch.stack(flows,dim=4), torch.stack([gt_flow]*len(flows),dim=4),torch.stack([mask]*len(flows),dim=4))
        # loss['hard L1 mask'], new_mask = L1_mask_hard_mining(torch.stack(flows,dim=4), torch.stack([gt_flow]*len(flows),dim=4),torch.stack([mask[:,0:1]]*len(flows),dim=4))
        # 
        # if args.loss=='epe':
        #     loss_total = loss['epe'] + loss['ld'] 
        # elif args.loss=='dfgvi':
        #     loss_total = loss['L1 mask'] + args.LAMBDA_HARD * loss['hard L1 mask'] 
        # elif args.loss=='l1':
        #     loss_total = loss['L1 mask']
        # elif args.loss=='mse':
        #     loss_total = loss['MSE mask']
        # elif args.loss=='l1_ld':
        #     loss_total = loss['L1 mask'] + args.lambda_ld*loss['ld']
        # elif args.loss=='mse_ld':
        #     loss_total = loss['MSE mask'] + args.lambda_ld*loss['ld']
        # 
        if args.GAN:
            l_f=[]
            l_gt=[]
        
            for b in range(mask.shape[0]):
                
                l_gt.append(torch.cat((mask[b,0:1,bbox[0][b]:bbox[0][b]+bbox[2][b],bbox[1][b]:bbox[1][b]+bbox[3][b]],input_x[b,3:,bbox[0][b]:bbox[0][b]+bbox[2][b],bbox[1][b]:bbox[1][b]+bbox[3][b]]),dim=0))
                fake_edges=edges[b] * mask[b,0:1,:,:] + input_x[b,3:,:,:] * (1. - mask[b,0:1,:,:])
                l_f.append(torch.cat((mask[b,0:1,bbox[0][b]:bbox[0][b]+bbox[2][b],bbox[1][b]:bbox[1][b]+bbox[3][b]],fake_edges[:,bbox[0][b]:bbox[0][b]+bbox[2][b],bbox[1][b]:bbox[1][b]+bbox[3][b]]),dim=0))


            f=torch.stack(l_f,dim=0)
            gt=torch.stack(l_gt,dim=0)
            
            Df=torch.mean(D(f))
            Dgt=torch.mean(D(gt))
            e=torch.rand(1).cuda()
            middle=e*gt+(1-e)*f
            D_middle=D(middle)
            grads = torch_grad(outputs=D_middle, inputs=middle,
                grad_outputs=torch.ones(D_middle.size()).cuda(),
                create_graph=True, retain_graph=True)[0]
            grad_loss=torch.mean((torch.norm(flat(grads),dim=1)-1)**2)
            
            loss['loss WpatchGAN D']=Df-Dgt
            
            optimizer_D.zero_grad()
            ((loss['loss WpatchGAN D']+1.*grad_loss)*args.lambda_gan).backward(retain_graph=True)
            optimizer_D.step()
            
            adjust_learning_rate(optimizer_D, i, steps)
            if i % args.n_critic==0:
                loss['loss WpatchGAN G']=torch.mean(-Df)
                loss_total= args.lambda_gan  *  loss['loss WpatchGAN G']
                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()
        
    
            
        
        
        
        
       
        
        
        adjust_learning_rate(optimizer, i, steps)


        if i % args.PRINT_EVERY == 0:
            
            #TESTING
            if not os.path.exists(log_dir+"/fake_edge"):
                os.makedirs(log_dir+"/fake_edge")
            # if not os.path.exists(log_dir+"/input"):
            #     os.makedirs(log_dir+"/input")            
            # if not os.path.exists(log_dir+"/output"):
            #     os.makedirs(log_dir+"/output")
            if not os.path.exists(log_dir+"/gt"):
                os.makedirs(log_dir+"/gt")
            # if not os.path.exists(log_dir+"/seg"):
            #     os.makedirs(log_dir+"/seg")
           
            # df=df.detach()
            # dfnp=torch.sigmoid(df).cpu().numpy()
            # Wdfnp=torch.sigmoid((df-df.mean(dim=0,keepdim=True))/(df.std(dim=0,keepdim=True)+10**-5)).cpu().numpy()
            # input=flow_masked[:,:2,:,:].detach().cpu().numpy()
            # gt=gt_flow[:,:,:,:].detach().cpu().numpy()
            # seg=input_x[:,3:,:,:].detach().cpu().numpy() #same name but this is actually the boundaries
            gt=input_x[:,3:,:,:].detach().cpu().numpy()
            fake_edge = (edges * mask[:,0:1,:,:] + input_x[:,3:,:,:] * (1. - mask[:,0:1,:,:])).detach().cpu().numpy()
          
            for j in range(flows[0].shape[0]):#batch
                for k in range(len(flows)):#layer
                    iio.write(log_dir+'/fake_edge/{:02d}_{:02d}.png'.format(j,k),fake_edge[j].transpose(1,2,0)*128+128)
                    iio.write(log_dir+'/gt/{:02d}_{:02d}.png'.format(j,k),gt[j].transpose(1,2,0)*128+128)
                   

            # print('=========================================================')
            # print(args.model_name, "Rank[{}] Iter [{}/{}],LR={}".format(0, i + 1, args.max_iter,args.LR))
            # print('=========================================================')
            # print_loss_dict(loss)
            
            write_loss_dict(loss, writer, i)

        if (i+1) % args.MODEL_SAVE_STEP == 0:
            save_ckpt(os.path.join(model_save_dir, 'DFI_%d.pth' % i),
                      [('model', IN)], [('optimizer', optimizer)], i)
            print('Model has been saved at %d Iters' % i)
    writer.close()


if __name__ == '__main__':
    main()

    
    
    
    
    
    
    
    
    
    
    
    
    
