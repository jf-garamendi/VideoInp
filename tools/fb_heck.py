##
# args.edge=None
# args.image=None
# args.rflow=True
# args.n_frames=6
# args.two_masks=True
# def test():
#     global l
#     train_dataset = dataset(args)
#     train_loader = DataLoader(train_dataset,
#                                 batch_size=args.batch_size,
#                                 shuffle=True,
#                                 drop_last=True,
#                                 num_workers=0)
#     train_iterator = iter(train_loader)
#     for i in range(2):
#         print(i)
#     
#         try:
#         
#             l = next(train_iterator)
#             for e in l[:-1]:
#                 if e==None:
#                     print(None)
#                 else:
#                     print(e.shape)
#         except:
#             print('Loader Restart')
#             l = next(train_iterator)
#             for e in l[:-1]:
#                 if e==None:
#                     print(None)
#                 else:
#                     print(e.shape)
##

i=3
b=l[4][i,3]
f=l[3][i,2]
xx,yy=torch.meshgrid(torch.arange(f.shape[1]),torch.arange(f.shape[2]))
indx=xx.reshape(-1)
indy=yy.reshape(-1)
dx,dy=f[:,indx,indy]
warped_indx=torch.clamp(indx+dx,0,b.shape[1]-1.01)
warped_indy=torch.clamp(indy+dy,0,b.shape[2]-1.01)
decimal_x=warped_indx-warped_indx.int()
decimal_y=warped_indy-warped_indy.int()

interp= b[:,warped_indx.int().type(torch.LongTensor),warped_indy.int().type(torch.LongTensor)]*(1-decimal_x)*(1-decimal_y)+b[:,warped_indx.int().type(torch.LongTensor)+1,warped_indy.int().type(torch.LongTensor)]*decimal_x*(1-decimal_y)+b[:,warped_indx.int().type(torch.LongTensor),warped_indy.int().type(torch.LongTensor)+1]*decimal_y*(1-decimal_x)+b[:,warped_indx.int().type(torch.LongTensor)+1,warped_indy.int().type(torch.LongTensor)+1]*decimal_x*decimal_y


d=torch.norm(interp+f[:,indx,indy],dim=0)

##
import time


b=l[4][:,3]
f=l[3][:,2]

tic=time.time()
bb,xx,yy=torch.meshgrid(torch.arange(f.shape[0]),torch.arange(f.shape[-2]),torch.arange(f.shape[-1]))
indb=bb.reshape(-1)
indx=xx.reshape(-1)
indy=yy.reshape(-1)
dx=f[indb,0,indx,indy]
dy=f[indb,1,indx,indy]
warped_indx=torch.clamp(indx+dx,0,b.shape[-2]-1.01)
warped_indy=torch.clamp(indy+dy,0,b.shape[-1]-1.01)
decimal_x=(warped_indx-warped_indx.int()).unsqueeze(-1)
decimal_y=(warped_indy-warped_indy.int()).unsqueeze(-1)

interp= b[indb,:,warped_indx.int().type(torch.LongTensor),warped_indy.int().type(torch.LongTensor)]*(1-decimal_x)*(1-decimal_y)+b[indb,:,(warped_indx.int()+1).type(torch.LongTensor),warped_indy.int().type(torch.LongTensor)]*decimal_x*(1-decimal_y)+b[indb,:,warped_indx.int().type(torch.LongTensor),(warped_indy.int()+1).type(torch.LongTensor)]*decimal_y*(1-decimal_x)+b[indb,:,(warped_indx.int()+1).type(torch.LongTensor),(warped_indy.int()+1).type(torch.LongTensor)]*decimal_x*decimal_y
d=torch.norm(interp+f[indb,:,indx,indy],dim=1)
dd=d.reshape(32,240,424)
t1=time.time()-tic

##
tic=time.time()
xx,yy=torch.meshgrid(torch.arange(f.shape[-2]),torch.arange(f.shape[-1]))
ind=torch.stack((yy,xx),dim=-1)
ind=ind.repeat(f.shape[0],1,1,1)
grid=f.permute((0,2,3,1))+ind
grid=(2*grid/torch.tensor([f.shape[-1]*1.,f.shape[-2]*1.]).view(1,1,1,2))-1

interp=torch.nn.functional.grid_sample(b,grid,mode='bilinear',padding_mode='border',align_corners=False)
d=torch.norm(interp+f,dim=1)
t2=time.time()-tic
print(d.shape)
##
plt.figure()
i=4
vm=max(dd[i].max(),d[i].max())
plt.subplot(1,2,1)
plt.title('forward backward error')
plt.imshow(dd[i],vmax=vm)
plt.title('hand made interpolation, t={:f}'.format(t1))
plt.colorbar()
plt.subplot(1,2,2)
plt.title('grid_sample interpolation, t={:f}'.format(t2))
plt.imshow(d[i],vmax=vm)
plt.colorbar()

plt.show()

##

plt.figure()
for i in range(32):
    print(d[i].mean())
    plt.subplot(8,4,i+1)
    plt.imshow(d[i])
    plt.colorbar()
plt.show()
##
i=7

fi=f[i]
bi=b[i]
ddi=dd[i]
di=d[i]
#di=(interp+f[indb,:,indx,indy])[:,0].reshape(32,-1)[i]
plt.figure()
plt.set_cmap('hot')
plt.subplot(321)
plt.imshow(ddi)
plt.title('old error')
plt.colorbar()
plt.subplot(322)
plt.imshow(di)
plt.title('new error')
plt.colorbar()
plt.subplot(323)
plt.title('x forward flow')
plt.imshow(fi[0])
plt.colorbar()
plt.subplot(324)
plt.title('y forward flow')
plt.imshow(fi[1])
plt.colorbar()
plt.subplot(325)
plt.title('x backward flow')
plt.imshow(bi[0])
plt.colorbar()
plt.subplot(326)
plt.title('y backward flow')
plt.imshow(bi[1])
plt.colorbar()
plt.show()
