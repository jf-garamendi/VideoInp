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

args = parser.parse_args()

def main():
    print(args.model_name)
    flat=Flatten()
    steps=[1,2,3]
    
    in_c=args.in_c
    out_c=args.out_c
    
    print(in_c,out_c)
    
    if args.net=='interponet':
        IN=InterpoNet()
    elif args.net=='res':
        IN=Res(in_c,out_c,10,0)
    elif args.net=='unet':
        IN=UNet(in_c,out_c)
        
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
        
    if args.patchGAN:
       
        D=Patch_Discriminator(30).cuda().train()
        optimizer_D = optim.Adam(D.parameters(),lr=args.LR,betas=(args.BETA1,args.BETA2))
    elif args.GAN:
       
        D=Discriminator(30).cuda().train()
        optimizer_D = optim.Adam(D.parameters(),lr=args.LR,betas=(args.BETA1,args.BETA2))
   
    if args.patchGANmask or args.WpatchGANmask:
        D=Discriminator_mask(5).cuda().train()
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
    it = 0 if not args.resume else resume_iter

    for epoch in range(args.n_epochs):
        
        print('starting epoch {:d}'.format(epoch))
        adjust_learning_rate(optimizer, epoch, steps)
        for i, (flow_masked,rflow_masked, mask, gt_all_flow, gt_all_rflow, edge, img, bbox) in enumerate(tqdm(train_loader)):
            it+=1
            # try:
            #     = next(train_iterator)
            # except:
            #     print('Loader Restart')
            #     train_iterator=0
            #     train_iterator=iter(train_loader)
            #     print('flag0')
            #     flow_masked,rflow_masked, mask, gt_flow, gt_rflow, edge, img, bbox = next(train_iterator)
            #     print('flag1')
            # print(time.time()-st)
            gt_all_rflow=gt_all_rflow.cuda()
            gt_all_flow=gt_all_flow.cuda()
            B,N,C,H,W=flow_masked.shape
            input_x =gt_all_flow[:,0].cuda()
            if args.seg=='b':
                gt_warp= gt_all_rflow[:,1].cuda()
            elif args.seg=='f':
                gt_warp= gt_all_flow[:,1].cuda()
            
          
    
            flows=IN(input_x)
         
            #fflows = cat (dim=1) forward flow [2], forward_flow[3]
            
            #loss['ld'] = ld(torch.stack(flows,dim=4), torch.stack([gt_flow]*len(flows),dim=4))
            loss['L1 mask']= L1_mask(torch.stack(flows,dim=4), torch.stack([gt_warp]*len(flows),dim=4))
            
            #loss['MSE mask'] = MSE_mask(torch.stack(flows,dim=4), torch.stack([gt_flow]*len(flows),dim=4),torch.stack([mask]*len(flows),dim=4))
            #loss['hard L1 mask'], new_mask = L1_mask_hard_mining(torch.stack(flows,dim=4), torch.stack([gt_flow]*len(flows),dim=4),torch.stack([mask[:,0:1]]*len(flows),dim=4))
            
            #grad_norm_squared=torch.sum((flows[0][:,:,1:,:-1]-flows[0][:,:,:-1,:-1])**2 + (flows[0][:,:,:-1,1:]-flows[0][:,:,:-1,:-1])**2,dim=1,keepdim=True)
            
            #loss['grad']=torch.abs(grad_norm_squared*mask[:,0:1,:-1,:-1]*(1-edges[:,:,:-1masked,:-1])).sum()/mask[:,0:1].sum()
            
            loss_total=0.
            
            if args.patchGAN or args.GAN :
                l_fake=[]
                l_gt=[]
                
    
                for b in range(mask.shape[0]):
  
                    # l_gt.append(torch.cat((fmask[b,0:1,bbox[0][b]:bbox[0][b]+bbox[2][b],bbox[1][b]:bbox[1][b]+bbox[3][b]],bmask[b,0:1,bbox[0][b]:bbox[0][b]+bbox[2][b],bbox[1][b]:bbox[1][b]+bbox[3][b]],gt_flow[b,:,bbox[0][b]:bbox[0][b]+bbox[2][b],bbox[1][b]:bbox[1][b]+bbox[3][b]],gt_rflow[b,:,bbox[0][b]:bbox[0][b]+bbox[2][b],bbox[1][b]:bbox[1][b]+bbox[3][b]]),dim=0))
                    # fake=torch.cat((fflows[0][b] * fmask[b,:,:,:] + flow_masked[b,2,:,:,:] * (1. - fmask[b,:,:,:]),bflows[0][b] * bmask[b,:,:,:] + rflow_masked[b,3,:,:,:] * (1. - bmask[b,:,:,:]) ),dim=0)
                
   #                #   l_fake.append(torch.cat((fmask[b,0:1,bbox[0][b]:bbox[0][b]+bbox[2][b],bbox[1][b]:bbox[1][b]+bbox[3][b]],bmask[b,0:1,bbox[0][b]:bbox[0][b]+bbox[2][b],bbox[1][b]:bbox[1][b]+bbox[3][b]],fake[:,bbox[0][b]:bbox[0][b]+bbox[2][b],bbox[1][b]:bbox[1][b]+bbox[3][b]]),dim=0))
                 
                    l_gt.append(torch.cat((gt_all_flow[b].view(N*C,H,W).cuda(),gt_all_rflow[b].view(N*C,H,W).cuda(),mask[b,:,0]),dim=0)[:,bbox[0][b]:bbox[0][b]+bbox[2][b],bbox[1][b]:bbox[1][b]+bbox[3][b]])
                    
                    fake_batch_flow=gt_all_flow[b].cuda()*1.
                    fake_batch_rflow=gt_all_rflow[b].cuda()*1.
                    
                    fake_batch_flow[2]= flows[0][b,:2]* fmask[b,:,:,:] + flow_masked[b,2,:,:,:] * (1. - fmask[b,:,:,:])
                    fake_batch_flow[3]= flows[0][b,2:4]* bmask[b,:,:,:] + flow_masked[b,2,:,:,:] * (1. - bmask[b,:,:,:])
                    fake_batch_rflow[2]=flows[0][b,4:6] * fmask[b,:,:,:] + rflow_masked[b,3,:,:,:] * (1. - fmask[b,:,:,:]) 
                    fake_batch_rflow[3]=flows[0][b,6:8] * bmask[b,:,:,:] + rflow_masked[b,3,:,:,:] * (1. - bmask[b,:,:,:]) 
                
                    l_fake.append(torch.cat((fake_batch_flow.view(N*C,H,W),fake_batch_rflow.view(N*C,H,W),mask[b,:,0]),dim=0)[:,bbox[0][b]:bbox[0][b]+bbox[2][b],bbox[1][b]:bbox[1][b]+bbox[3][b]])
                    
    
    
    
    
                fake=torch.stack(l_fake,dim=0)
                gt=torch.stack(l_gt,dim=0)
                
                Df=torch.mean(D(fake))
                Dgt=torch.mean(D(gt))
                e=torch.rand(1).cuda()
                middle=e*gt+(1-e)*fake
                D_middle=D(middle)
                grads = torch_grad(outputs=D_middle, inputs=middle,
                    grad_outputs=torch.ones(D_middle.size()).cuda(),
                    create_graph=True, retain_graph=True)[0]
                grad_loss=torch.mean((torch.norm(flat(grads),dim=1)-1)**2)
                
                loss['loss WpatchGAN D']=(Df-Dgt).detach()
                
                optimizer_D.zero_grad()
                ((Df-Dgt+0.1*grad_loss)*args.lambda_gan).backward(retain_graph=True)
                optimizer_D.step()
                
                adjust_learning_rate(optimizer_D, epoch, steps)
                if i % args.n_critic==0:
                    loss['loss WpatchGAN G']=torch.mean(-Df).detach()
                    loss_total+= args.lambda_gan  *  torch.mean(-Df)
            #todo discrimator for edges
            
            if args.multipatchGAN:
                
    
                
                fake = torch.cat((flows[0] * mask[:,:,:,:] + flow_masked[:,:,:,:] * (1. - mask[:,:,:,:]),mask[:,0:1]),dim=1)
                gt = torch.cat((gt_flow,mask[:,0:1]),dim=1) 
                
                Df=0
                Dgt=0
                df=D(fake)
                dgt=D(gt)
                for j in range(3):
                    Df+=torch.mean(df[j])/3
                    Dgt+=torch.mean(dgt[j])/3
                
                e=torch.rand(1).cuda()
                middle=e*gt+(1-e)*fake
                D_middle=D(middle)
                grad_loss=0
                for j in range(3):
                    grads = torch_grad(outputs=D_middle[j], inputs=middle,
                        grad_outputs=torch.ones(D_middle[j].size()).cuda(),
                        create_graph=True, retain_graph=True)[0]
                    grad_loss+=torch.mean((torch.norm(flat(grads),dim=1)-1)**2)/3
                
                loss['loss WpatchGAN D']=Df-Dgt
                
                optimizer_D.zero_grad()
                ((loss['loss WpatchGAN D']+1.*grad_loss)*args.lambda_gan).backward(retain_graph=(i % args.n_critic==0))
                optimizer_D.step()
                
                
                adjust_learning_rate(optimizer_D, epoch, steps)
            
                if i % args.n_critic==0:
                    loss['loss WpatchGAN G']=torch.mean(-Df)
                    loss_total+= args.lambda_gan  *  loss['loss WpatchGAN G']
            
            
    
        
            
            
            #(fflows[0][:,2:] * mask[:,3,:,:] + flow_masked[:,3,:,:,:] * (1. - mask[:,3,:,:])) # fflows[0][:,2:]
            #(bflows[0][:,:2] * mask[:,2,:,:] + rflow_masked[:,2,:,:,:] * (1. - mask[:,2,:,:]))
            
            # loss['forward-backward']=.5*(fb(fflows[0][:,:2]*mask[:,2] + flow_masked[:,2] * (1. - mask[:,2]),bflows[0][:,2:]*mask[:,3] + rflow_masked[:,3]*(1. - mask[:,3]))+fb(bflows[0][:,2:]*mask[:,3] + rflow_masked[:,3]*(1. - mask[:,3]),fflows[0][:,:2]*mask[:,2] + flow_masked[:,2] * (1. - mask[:,2])))
            # loss['forward-backward gt-pred']=.25*(fb(
            # fflows[0][:,2:]*mask[:,3] + flow_masked[:,3] * (1. - mask[:,3]),gt_all_rflow[:,4])+fb(
            # gt_all_rflow[:,4],fflows[0][:,2:]*mask[:,3] + flow_masked[:,3] * (1. - mask[:,3]))+fb((bflows[0][:,:2]*mask[:,2] + rflow_masked[:,2]*(1. - mask[:,2])),gt_all_flow[:,1]))
            if args.loss=='l1':
                loss_total += loss['L1 mask']*1.
            elif args.loss=='l1_ld':
                loss_total += loss['L1 mask']*1. + args.lambda_ld*loss['ld']
            elif args.loss=='l1_fb':
                loss_total += loss['L1 mask']*1. + args.lambda_fb*(.5*loss['forward-backward']+loss['forward-backward gt-pred'])
            elif args.loss=='fb':
                loss_total += args.lambda_fb*(.5*loss['forward-backward']+loss['forward-backward gt-pred'])
                
            elif args.loss=='none':
                loss_total +=loss['L1 mask']*0.
                
                
            
                
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            
            for n,e in loss.items():
                e=e.detach()
    
    
        
    
            write_loss_dict(loss, writer, it)
           
    
    
            if i % args.PRINT_EVERY == 0:
    
                #TESTING
                if not os.path.exists(log_dir+"/forward"):
                    os.makedirs(log_dir+"/forward")
                if not os.path.exists(log_dir+"/backward"):
                    os.makedirs(log_dir+"/backward")            
                if not os.path.exists(log_dir+"/output"):
                    os.makedirs(log_dir+"/output")

    

                for j in range(flows[0].shape[0]):#batch
                   
                    
                    iio.write(log_dir+'/forward/{:02d}.flo'.format(j),input_x[j].detach().cpu().numpy().transpose(1,2,0))
                    iio.write(log_dir+'/backward/{:02d}.flo'.format(j),gt_warp[j].detach().cpu().numpy().transpose(1,2,0))
                    iio.write(log_dir+'/output/{:02d}.flo'.format(j),flows[0][j].detach().cpu().numpy().transpose(1,2,0))
                        
    
                # print('=========================================================')
                # print(args.model_name, "Rank[{}] Iter [{}/{}],LR={}".format(0, i + 1, args.max_iter,args.LR))
                # print('=========================================================')
                #print_loss_dict(loss)
                # IN.eval()
                # val_iterator = iter(val_loader)
                # loss['epe_val']=0
                # with torch.no_grad():
                #     for j in range(0,20):
                #         flow_mask_cat, _,gt_flow,_,_ = next(val_iterator)
                #         input_x = flow_mask_cat.cuda()
                #         flows = IN(input_x
                
                
                #         loss['epe_val']+= epe(torch.stack(flows,dim=4), torch.stack([gt_flow.cuda()]*len(flows),dim=4))/20
                #         
    
                # IN.train()
               
    
            if (it+1) % args.MODEL_SAVE_STEP == 0:
                save_ckpt(os.path.join(model_save_dir, 'DFI_%d.pth' % it),
                        [('model', IN)], [('optimizer', optimizer)], it)
                print('Model has been saved at %d Iters' % it)
    writer.close()


if __name__ == '__main__':
    main()

    
    
    
    
    
    
    
    
    
    
    
    
    
