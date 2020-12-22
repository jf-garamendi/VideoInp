import torch

def ld(flows,gt_flow):
    w=0.5*torch.ones(flows.shape[-1])
    w[-1]=1
    w=w.cuda()
    
    out_x =flows[:,:,1:,:]-flows[:,:,:-1,:]
    gt_x =   gt_flow[:,:,1:,:]- gt_flow[:,:,:-1,:]
    out_y = flows[:,:,:,1:]-flows[:,:,:,:-1]
    gt_y =   gt_flow[:,:,:,1:]- gt_flow[:,:,:,:-1]
    epe_out_x=torch.sqrt(10**-10+torch.sum(out_x**2,dim=1))
    epe_gt_x=torch.sqrt(10**-10+torch.sum(gt_x**2,dim=1))
    epe_out_y=torch.sqrt(10**-10+torch.sum(out_y**2,dim=1))
    epe_gt_y=torch.sqrt(10**-10+torch.sum(gt_y**2,dim=1))
    maps=torch.abs(epe_out_x-epe_gt_x)[:,:,:-1]+torch.abs(epe_out_y-epe_gt_y)[:,:-1,:]
    return torch.mean(w*maps.mean(dim=0).mean(dim=0).mean(dim=0))
    
def epe(flows,gt_flow):
    w=0.5*torch.ones(flows.shape[-1])
    w[-1]=1
    w=w.cuda()
    
    return torch.mean(w*torch.norm(flows-gt_flow,dim=1).mean(dim=0).mean(dim=0).mean(dim=0))#norm along flow channels; mean along batch, x ,y; then wighted mean among layer of the net


def fb_old(f,b):
    bb,xx,yy=torch.meshgrid(torch.arange(f.shape[0]),torch.arange(f.shape[-2]),torch.arange(f.shape[-1]))
    indb=bb.reshape(-1).cuda()
    indx=xx.reshape(-1).cuda()
    indy=yy.reshape(-1).cuda()
    dx=f[indb,0,indx,indy]
    dy=f[indb,1,indx,indy]
    warped_indx=torch.clamp(indx+dx,0,b.shape[-2]-1.01)
    warped_indy=torch.clamp(indy+dy,0,b.shape[-1]-1.01)
    decimal_x=(warped_indx-warped_indx.int()).unsqueeze(-1)
    decimal_y=(warped_indy-warped_indy.int()).unsqueeze(-1)
    
    interp= b[indb,:,warped_indx.int().type(torch.LongTensor),warped_indy.int().type(torch.LongTensor)]*(1-decimal_x)*(1-decimal_y)+b[indb,:,(warped_indx.int()+1).type(torch.LongTensor),warped_indy.int().type(torch.LongTensor)]*decimal_x*(1-decimal_y)+b[indb,:,warped_indx.int().type(torch.LongTensor),(warped_indy.int()+1).type(torch.LongTensor)]*decimal_y*(1-decimal_x)+b[indb,:,(warped_indx.int()+1).type(torch.LongTensor),(warped_indy.int()+1).type(torch.LongTensor)]*decimal_x*decimal_y
    d=torch.norm(interp+f[indb,:,indx,indy],dim=1)
    return d.mean()

def fb(f,b,test=False):
    xx,yy=torch.meshgrid(torch.arange(f.shape[-2]),torch.arange(f.shape[-1]))
    ind=torch.stack((yy,xx),dim=-1)
    ind=ind.repeat(f.shape[0],1,1,1).cuda()
    grid=f.permute((0,2,3,1))+ind
    grid=(2*grid/torch.tensor([f.shape[-1]*1.,f.shape[-2]*1.]).cuda().view(1,1,1,2))-1
    
    interp=torch.nn.functional.grid_sample(b,grid,mode='bilinear',padding_mode='border',align_corners=False)
    d=torch.norm(interp+f,dim=1)
    if test:
        return d
    else:
        return d.mean()








def MSE_mask(x, y, mask=None):
    res = x - y
    if mask is not None:
        res = res[mask==1]
        return torch.mean(res**2)
        
    return torch.mean(res**2)



def L1(x, y, mask=None):
    res = torch.abs(x - y)
    if mask is not None:
        res = res * mask
        
    return torch.mean(res)


def L1_mask(x, y, mask=None):
    res = torch.abs(x - y)

    if mask is not None:
        return res[mask==1].mean()
        #res = res * mask
        #return torch.sum(res) / torch.sum(mask)
    return torch.mean(res)


def L1_mask_hard_mining(x, y, mask):

    input_size = x.size()
    res = torch.sum(torch.abs(x - y), dim=1, keepdim=True)
    
    with torch.no_grad():
        idx = mask > 0.5
        res_sort = [torch.sort(res[i, idx[i, ...]])[0] for i in range(idx.shape[0])]
        res_sort = [  i[int(i.shape[0] * 0.5)].item() if len(i)>0 else 0 for i in res_sort ]#.5=threshold of 50 percent
        new_mask = mask.clone()
        for i in range(res.shape[0]):
            new_mask[i, ...] = ((mask[i, ...] > 0.5) & (res[i, ...] > res_sort[i])).float()

    res = res * new_mask
    final_res = torch.sum(res) / torch.sum(new_mask)
    return final_res, new_mask
