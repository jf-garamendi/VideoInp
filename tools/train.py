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
from utils.data import *
from utils.losses import *
from utils.io import *
parser = argparse.ArgumentParser()

Lap=Laplacian(3)

# training options
parser.add_argument('--net', type=str, default='unet')
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
    elif args.joint:
        in_c=4
    else:
        in_c =4
    
    if args.joint:
        out_c=3
    else:
        out_c =2
    
    print(in_c,out_c)
    
    if args.net=='interponet':
        IN=InterpoNet()
    elif args.net=='res':
        IN=Res(in_c,10,10)
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
        if not args.joint:
            D=Patch_Discriminator(3).cuda().train()
        else:
            D=Patch_Discriminator(4).cuda().train()
        optimizer_D = optim.Adam(D.parameters(),lr=args.LR,betas=(args.BETA1,args.BETA2))
    elif args.GAN:
        if not args.joint:
            D=Discriminator(3).cuda().train()
        else:
            D=Discriminator(4).cuda().train()
        optimizer_D = optim.Adam(D.parameters(),lr=args.LR,betas=(args.BETA1,args.BETA2))
        
    if args.multipatchGAN:
        D=multi_Discriminator(3).cuda().train()
        optimizer_D = optim.Adam(D.parameters(),lr=args.LR,betas=(args.BETA1,args.BETA2))
    if args.patchGANmask or args.WpatchGANmask:
        D=Discriminator_mask(2).cuda().train()
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

        try:
            flow_mask_cat, flow_masked, gt_flow, mask, bbox = next(train_iterator)
        except:
            print('Loader Restart')
            train_iterator = iter(train_loader)
            flow_mask_cat, flow_masked, gt_flow, mask, bbox = next(train_iterator)

        # print(time.time()-st)
        input_x = flow_mask_cat.cuda()
        gt_flow = gt_flow.cuda()
        mask = mask.cuda()
        flow_masked = flow_masked.cuda()

        if args.joint:
            input_x_masked_edge=input_x*1.
            input_x_masked_edge[:,3:] = input_x[:,3:,:,:] * (1. - mask[:,0:1,:,:])
            flows=IN(input_x_masked_edge)
            flows, edges = [flows[0][:,:2]],torch.tanh(flows[0][:,2:3])
        else:
            flows=IN(input_x)
        
        loss['epe']= epe(torch.stack(flows,dim=4), torch.stack([gt_flow]*len(flows),dim=4))
        loss['ld'] = ld(torch.stack(flows,dim=4), torch.stack([gt_flow]*len(flows),dim=4))
        
        loss['L1 mask'] = L1_mask(torch.stack(flows,dim=4), torch.stack([gt_flow]*len(flows),dim=4),torch.stack([mask]*len(flows),dim=4))
        if loss['L1 mask']<0:
            print(loss['L1 mask'])
        loss['MSE mask'] = MSE_mask(torch.stack(flows,dim=4), torch.stack([gt_flow]*len(flows),dim=4),torch.stack([mask]*len(flows),dim=4))
        loss['hard L1 mask'], new_mask = L1_mask_hard_mining(torch.stack(flows,dim=4), torch.stack([gt_flow]*len(flows),dim=4),torch.stack([mask[:,0:1]]*len(flows),dim=4))
        
        grad_norm_squared=torch.sum((flows[0][:,:,1:,:-1]-flows[0][:,:,:-1,:-1])**2 + (flows[0][:,:,:-1,1:]-flows[0][:,:,:-1,:-1])**2,dim=1,keepdim=True)
        
        loss['grad']=torch.abs(grad_norm_squared*mask[:,0:1,:-1,:-1]*(1-edges[:,:,:-1,:-1])).sum()/mask[:,0:1].sum()
        
        if args.loss=='epe':
            loss_total = loss['epe'] + loss['ld'] 
        elif args.loss=='dfgvi':
            loss_total = loss['L1 mask'] + args.LAMBDA_HARD * loss['hard L1 mask'] 
        elif args.loss=='l1':
            loss_total = loss['L1 mask']*1.
        elif args.loss=='mse':
            loss_total = loss['MSE mask']*1.
        elif args.loss=='l1_ld':
            loss_total = loss['L1 mask'] + args.lambda_ld*loss['ld']
        elif args.loss=='mse_ld':
            loss_total = loss['MSE mask'] + args.lambda_ld*loss['ld']
        elif args.loss=='consistency':
            loss_total=  loss['L1 mask']+args.lambda_consistency*loss['grad']
        elif args.loss=='none':
            loss_total=0.
            
        
        if args.patchGAN or args.GAN :
            l_f=[]
            l_gt=[]
            
            if args.joint:
                for b in range(mask.shape[0]):
                    
                    l_gt.append(torch.cat((mask[b,0:1,bbox[0][b]:bbox[0][b]+bbox[2][b],bbox[1][b]:bbox[1][b]+bbox[3][b]],gt_flow[b,:,bbox[0][b]:bbox[0][b]+bbox[2][b],bbox[1][b]:bbox[1][b]+bbox[3][b]],input_x[b,3:,bbox[0][b]:bbox[0][b]+bbox[2][b],bbox[1][b]:bbox[1][b]+bbox[3][b]]),dim=0))
                    fake=flows[0][b] * mask[b,:,:,:] + flow_masked[b,:,:,:] * (1. - mask[b,:,:,:])
                    fake_edges=edges[b] * mask[b,0:1,:,:] + input_x[b,3:,:,:] * (1. - mask[b,0:1,:,:])
                    
                    l_f.append(torch.cat((mask[b,0:1,bbox[0][b]:bbox[0][b]+bbox[2][b],bbox[1][b]:bbox[1][b]+bbox[3][b]],fake[:,bbox[0][b]:bbox[0][b]+bbox[2][b],bbox[1][b]:bbox[1][b]+bbox[3][b]],fake_edges[:,bbox[0][b]:bbox[0][b]+bbox[2][b],bbox[1][b]:bbox[1][b]+bbox[3][b]]),dim=0))
                    
                    
            else:
                for b in range(mask.shape[0]):
                    # l_gt.append(torch.cat((flow_masked[b,:,bbox[0][b]:bbox[0][b]+bbox[2][b],bbox[1][b]:bbox[1][b]+bbox[3][b]],gt_flow[b,:,bbox[0][b]:bbox[0][b]+bbox[2][b],bbox[1][b]:bbox[1][b]+bbox[3][b]]),dim=0))
                    # fake=flows[0][b] * mask[b,:,:,:] + flow_masked[b,:,:,:] * (1. - mask[b,:,:,:])
                    # l_f.append(torch.cat((flow_masked[b,:,bbox[0][b]:bbox[0][b]+bbox[2][b],bbox[1][b]:bbox[1][b]+bbox[3][b]],fake[:,bbox[0][b]:bbox[0][b]+bbox[2][b],bbox[1][b]:bbox[1][b]+bbox[3][b]]),dim=0))
                   
                    l_gt.append(torch.cat((mask[b,0:1,bbox[0][b]:bbox[0][b]+bbox[2][b],bbox[1][b]:bbox[1][b]+bbox[3][b]],gt_flow[b,:,bbox[0][b]:bbox[0][b]+bbox[2][b],bbox[1][b]:bbox[1][b]+bbox[3][b]]),dim=0))
                    fake=flows[0][b] * mask[b,:,:,:] + flow_masked[b,:,:,:] * (1. - mask[b,:,:,:])
                    l_f.append(torch.cat((mask[b,0:1,bbox[0][b]:bbox[0][b]+bbox[2][b],bbox[1][b]:bbox[1][b]+bbox[3][b]],fake[:,bbox[0][b]:bbox[0][b]+bbox[2][b],bbox[1][b]:bbox[1][b]+bbox[3][b]]),dim=0))



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
                loss_total+= args.lambda_gan  *  loss['loss WpatchGAN G']
        #todo discrimator for edges
        
        if args.multipatchGAN:
            

            
            f = torch.cat((flows[0] * mask[:,:,:,:] + flow_masked[:,:,:,:] * (1. - mask[:,:,:,:]),mask[:,0:1]),dim=1)
            gt = torch.cat((gt_flow,mask[:,0:1]),dim=1) 
            
            Df=0
            Dgt=0
            df=D(f)
            dgt=D(gt)
            for j in range(3):
                Df+=torch.mean(df[j])/3
                Dgt+=torch.mean(dgt[j])/3
            
            
            
            
            e=torch.rand(1).cuda()
            middle=e*gt+(1-e)*f
            D_middle=D(middle)
            grad_loss=0
            for j in range(3):
                grads = torch_grad(outputs=D_middle[j], inputs=middle,
                    grad_outputs=torch.ones(D_middle[j].size()).cuda(),
                    create_graph=True, retain_graph=True)[0]
                grad_loss+=torch.mean((torch.norm(flat(grads),dim=1)-1)**2)/3
            
            loss['loss WpatchGAN D']=Df-Dgt
            
            optimizer_D.zero_grad()
            ((loss['loss WpatchGAN D']+1.*grad_loss)*args.lambda_gan).backward(retain_graph=True)
            optimizer_D.step()
            
            
            adjust_learning_rate(optimizer_D, i, steps)
        
            if i % args.n_critic==0:
                loss['loss WpatchGAN G']=torch.mean(-Df)
                loss_total+= args.lambda_gan  *  loss['loss WpatchGAN G']
        
        
        
        if args.patchGANmask:
            l_f=[]
            l_gt=[]
            l_mask_gt=[]
            for b in range(mask.shape[0]):
                l_gt.append(gt_flow[b,:,bbox[0][b]:bbox[0][b]+bbox[2][b],bbox[1][b]:bbox[1][b]+bbox[3][b]])
                fake=flows[0][b] * mask[b,:,:,:] + flow_masked[b,:,:,:] * (1. - mask[b,:,:,:])
                l_f.append(fake[:,bbox[0][b]:bbox[0][b]+bbox[2][b],bbox[1][b]:bbox[1][b]+bbox[3][b]])
                l_mask_gt.append(mask[b,0:1,bbox[0][b]:bbox[0][b]+bbox[2][b],bbox[1][b]:bbox[1][b]+bbox[3][b]])
                
            f=torch.stack(l_f,dim=0)
            gt=torch.stack(l_gt,dim=0)
            mask_gt=torch.stack(l_mask_gt,dim=0)
            
            df=D(f)
            dgt=D(gt)
            m=mask_gt.mean()
   
            loss['loss WpatchGAN D']=nn.BCEWithLogitsLoss(reduction='mean')(torch.cat((df,dgt),dim=0),torch.cat((mask_gt,0.*mask_gt),dim=0))

            optimizer_D.zero_grad()
            (loss['loss WpatchGAN D']*args.lambda_gan).backward(retain_graph=True)
            optimizer_D.step()
            adjust_learning_rate(optimizer_D, i, steps)
            
            loss['loss WpatchGAN G']=-loss['loss WpatchGAN D']
            loss_total+= args.lambda_gan  *  loss['loss WpatchGAN G']
        
            
        if args.WpatchGANmask:
            l_f=[]
            l_gt=[]
            l_mask_gt=[]
            grad_loss_index=[]
            for b in range(mask.shape[0]):
                
               
                l_gt.append(gt_flow[b,:,bbox[0][b]:bbox[0][b]+bbox[2][b],bbox[1][b]:bbox[1][b]+bbox[3][b]])
                fake=flows[0][b] * mask[b,:,:,:] + flow_masked[b,:,:,:] * (1. - mask[b,:,:,:])
                l_f.append(fake[:,bbox[0][b]:bbox[0][b]+bbox[2][b],bbox[1][b]:bbox[1][b]+bbox[3][b]])
                l_mask_gt.append(mask[b,0:1,bbox[0][b]:bbox[0][b]+bbox[2][b],bbox[1][b]:bbox[1][b]+bbox[3][b]])
                
                index=[np.random.randint(bbox[2][b]),np.random.randint(bbox[3][b])]
                while l_mask_gt[-1][0,index[0],index[1]]==0:
                    index=[np.random.randint(bbox[2][b]),np.random.randint(bbox[3][b])]
                grad_loss_index.append(torch.tensor(index).cuda())
                 
            f=torch.stack(l_f,dim=0)
            gt=torch.stack(l_gt,dim=0)
            mask_gt=torch.stack(l_mask_gt,dim=0)
            index=torch.stack(grad_loss_index,dim=0)
            df=D(f)
            dgt=D(gt)
            m=mask_gt.mean()
            
            Df=torch.mean(df*mask_gt)/m
            Dgt=(torch.mean(df*(1.-mask_gt))+torch.mean(dgt))/(2-m)
            
            e=torch.rand(1).cuda()
            middle=e*f+(1-e)*gt
            #D_middle=(D(middle)[:,0,index[:,0],index[:,1]]).diag()
            D_middle=D(middle)
            optimizer_D.zero_grad()
            idx,idy=torch.randint(middle.shape[2],(args.batch_size,)),torch.randint(middle.shape[3],(args.batch_size,))
            point=(D_middle[:,0,idx,idy]).diag()
            grads = torch_grad(outputs=point,inputs=middle,grad_outputs=torch.ones(point.size()).cuda(),create_graph=True,retain_graph=True)[0]
            grad_loss=(torch.mean((torch.norm(flat(grads),dim=1)-1)**2))
            
            loss['loss WpatchGAN D']=Df-Dgt
            
   
            
            ((loss['loss WpatchGAN D']+1.*grad_loss)*args.lambda_gan).backward(retain_graph=True)
            optimizer_D.step()
            
            
            
            adjust_learning_rate(optimizer_D, i, steps)
            
            if i % args.n_critic==0:
                loss['loss WpatchGAN G']=-Df
                loss_total += args.lambda_gan  *  loss['loss WpatchGAN G']
            
        
        
    
        if loss['L1 mask']<0:
            print('2',loss['L1 mask'])
        if loss_total != 0.:
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
        if loss['L1 mask']<0:
            print('3',loss['L1 mask'])
    
        
        
        adjust_learning_rate(optimizer, i, steps)


        if i % args.PRINT_EVERY == 0:
            print(loss['L1 mask'])
            #TESTING
            if not os.path.exists(log_dir+"/fake"):
                os.makedirs(log_dir+"/fake")
            if not os.path.exists(log_dir+"/input"):
                os.makedirs(log_dir+"/input")            
            if not os.path.exists(log_dir+"/output"):
                os.makedirs(log_dir+"/output")
            if not os.path.exists(log_dir+"/gt"):
                os.makedirs(log_dir+"/gt")
            if not os.path.exists(log_dir+"/seg"):
                os.makedirs(log_dir+"/seg")
            if not os.path.exists(log_dir+"/fake_edge"):
                os.makedirs(log_dir+"/fake_edge")
            if args.patchGANmask or args.WpatchGANmask:
                df=df.detach()
                dfnp=torch.sigmoid(df).cpu().numpy()
                Wdfnp=torch.sigmoid((df-df.mean(dim=0,keepdim=True))/(df.std(dim=0,keepdim=True)+10**-5)).cpu().numpy()
            input=flow_masked[:,:2,:,:].detach().cpu().numpy()
            gt=gt_flow[:,:,:,:].detach().cpu().numpy()
            seg=input_x[:,3:,:,:].detach().cpu().numpy() #same name but this is actually the boundaries
            for j in range(flows[0].shape[0]):#batch
                for k in range(len(flows)):#layer
                    fake = (flows[k] * mask[:,:,:,:] + flow_masked[:,:,:,:] * (1. - mask[:,:,:,:])).detach().cpu().numpy()
                    
                    
                    iio.write(log_dir+'/fake/{:02d}_{:02d}.flo'.format(j,k),fake[j].transpose(1,2,0))
                    iio.write(log_dir+'/gt/{:02d}_{:02d}.flo'.format(j,k),gt[j].transpose(1,2,0))
                    iio.write(log_dir+'/input/{:02d}_{:02d}.flo'.format(j,k),input[j].transpose(1,2,0))
                    iio.write(log_dir+'/seg/{:02d}_{:02d}.png'.format(j,k),seg[j].transpose(1,2,0)*128+128)
                    if args.joint:
                        fake_edge = (edges * mask[:,0:1,:,:] + input_x[:,3:,:,:] * (1. - mask[:,0:1,:,:])).detach().cpu().numpy()
                        
                        iio.write(log_dir+'/fake_edge/{:02d}_{:02d}.png'.format(j,k),fake_edge[j].transpose(1,2,0)*128+128)
                    if args.patchGANmask:
                        iio.write(log_dir+'/output/{:02d}_{:02d}.png'.format(j,k),(dfnp[j].transpose(1,2,0))*256)#flows[k][j].detach().cpu().numpy().transpose(1,2,0))
                    elif args.WpatchGANmask:
                        iio.write(log_dir+'/output/{:02d}_{:02d}.png'.format(j,k),(Wdfnp[j].transpose(1,2,0))*256)
                    else:
                        iio.write(log_dir+'/output/{:02d}_{:02d}.flo'.format(j,k),flows[k][j].detach().cpu().numpy().transpose(1,2,0))

            # print('=========================================================')
            # print(args.model_name, "Rank[{}] Iter [{}/{}],LR={}".format(0, i + 1, args.max_iter,args.LR))
            # print('=========================================================')
            print_loss_dict(loss)
            # IN.eval()
            # val_iterator = iter(val_loader)
            # loss['epe_val']=0
            # with torch.no_grad():
            #     for j in range(0,20):
            #         flow_mask_cat, _,gt_flow,_,_ = next(val_iterator)
            #         input_x = flow_mask_cat.cuda()
            #         flows = IN(input_x)
            #         loss['epe_val']+= epe(torch.stack(flows,dim=4), torch.stack([gt_flow.cuda()]*len(flows),dim=4))/20
            #         

            # IN.train()
            write_loss_dict(loss, writer, i)

        if (i+1) % args.MODEL_SAVE_STEP == 0:
            save_ckpt(os.path.join(model_save_dir, 'DFI_%d.pth' % i),
                      [('model', IN)], [('optimizer', optimizer)], i)
            print('Model has been saved at %d Iters' % i)
    writer.close()


if __name__ == '__main__':
    main()

    
    
    
    
    
    
    
    
    
    
    
    
    
