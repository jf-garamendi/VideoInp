import torch.nn as nn
import torch
from utils.partialconv2d import PartialConv2d
import torch.nn.functional as F

class WarpRes(nn.Module):
    def __init__(self,input_channels=4,output_channels=4, n_128=3,n_256=3):
        super(WarpRes, self).__init__()
        self.xx,self.yy=torch.meshgrid(torch.arange(240),torch.arange(424))
        self.l=[0]
        self.in_conv=nn.Conv2d(input_channels,64,7,1,3)
        
        self.L=nn.Sequential(nn.Conv2d(input_channels,64,3,3,0,dilation=2),
        nn.LeakyReLU(),
        nn.BatchNorm2d(64),
        Flatten(),
        nn.Linear(79*140*64,128),
        nn.BatchNorm1d(128),
        nn.LeakyReLU(),
        nn.Linear(128,2*64),
        nn.Tanh())
        
        self.convs=nn.ModuleList(
        [nn.Sequential(nn.Conv2d(128,64,3,1,1),nn.ELU(),
        ResBlock(64,64,64),
        nn.Conv2d(64,128,3,1,1))]+
        [nn.Sequential(ResBlock(128,64,128)) for i in range(n_128)]+
        # [nn.Conv2d(128,256,3,1,1)]+
        # [nn.Sequential(ResBlock(256,64,256)) for i in range(n_256)]+
         [nn.Conv2d(128,output_channels,3,1,1)])
            
        #self.detours=nn.ModuleList(
        # [nn.Sequential(nn.Conv2d(64,2,7,1,3)),
        # nn.Sequential(nn.Conv2d(64,2,7,1,3))]+
        # [nn.Sequential(nn.Conv2d(128,2,7,1,3)) for i in range(n_128+1)]+
        # [nn.Sequential(nn.Conv2d(256,2,7,1,3)) for i in range(n_256+1)])
    
    def check(self):
        return self.l


    def forward(self, x):
        out=self.in_conv(x)
        b,c,h,w=out.shape
        offset=self.L(x).view(b*c,1,1,2)*20
        # if x[0,0,0,0]<1 and x[0,0,0,0]>0:
        #    print(offset.view(b,c,-1)[0])
   
        ind=torch.stack((self.yy,self.xx),dim=-1).cuda()
        ind=ind.repeat(b*c,1,1,1).cuda()
        grid=ind+offset
        grid=(2*grid/torch.tensor([w,h]).cuda().view(1,1,1,2))-1
        out=nn.functional.leaky_relu(torch.cat((out,torch.nn.functional.grid_sample(out.view(b*c,1,h,w),grid,mode='bilinear',padding_mode='border',align_corners=False).view(b,c,h,w)),dim=1)  )
        self.l=offset.view(b,c,2)[0].detach()
        for i, l in enumerate(self.convs):
            out = self.convs[i](out)
            #out.append(self.detours[i](x))
        
        return [out]




























class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class Squeeze(nn.Module):
    def forward(self, x):
        return x.squeeze(1)

class Unsqueeze(nn.Module):
    def forward(self, x):
        return x.unsqueeze(1)


class InterpoNet(nn.Module):
    def __init__(self,input_channels=4, n_128=3,n_256=3):
        super(InterpoNet, self).__init__()
        self.convs=nn.ModuleList(
        [nn.Sequential(nn.Conv2d(input_channels,32,7,1,3),nn.ELU()),
        nn.Sequential(nn.Conv2d(32,64,7,1,3),nn.ELU()),
        nn.Sequential(nn.Conv2d(64,128,7,1,3),nn.ELU())]+
        [nn.Sequential(nn.Conv2d(128,128,7,1,3),nn.ELU()) for i in range(n_128)]+
        [nn.Sequential(nn.Conv2d(128,256,7,1,3),nn.ELU())]+
        [nn.Sequential(nn.Conv2d(256,256,7,1,3),nn.ELU()) for i in range(n_256)])
   
        self.detours=nn.ModuleList(
        [nn.Sequential(nn.Conv2d(32,2,7,1,3)),
        nn.Sequential(nn.Conv2d(64,2,7,1,3))]+
        [nn.Sequential(nn.Conv2d(128,2,7,1,3)) for i in range(n_128+1)]+
        [nn.Sequential(nn.Conv2d(256,2,7,1,3)) for i in range(n_256+1)])
        

    def forward(self, x):
        out=[]
        for i, l in enumerate(self.convs):
            x = self.convs[i](x)
            out.append(self.detours[i](x))
        return out







class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, outplanes):
        super(ResBlock, self).__init__()
        
        
        self.l=nn.Sequential(
        nn.Conv2d(inplanes,planes,kernel_size=3,padding=1,bias=False),
        nn.ELU(),
        nn.Conv2d(planes,outplanes,kernel_size=3,padding=1, bias=False),
        nn.BatchNorm2d(outplanes),
        )


    def forward(self, x):
        return x+self.l(x)



class Res(nn.Module):
    def __init__(self,input_channels=4,output_channels=4, n_128=3,n_256=3):
        super(Res, self).__init__()
        self.convs=nn.ModuleList(
        [nn.Sequential(nn.Conv2d(input_channels,64,7,1,3),nn.ELU(),
        ResBlock(64,64,64),
        nn.Conv2d(64,128,3,1,1))]+
        [nn.Sequential(ResBlock(128,64,128)) for i in range(n_128)]+
        # [nn.Conv2d(128,256,3,1,1)]+
        # [nn.Sequential(ResBlock(256,64,256)) for i in range(n_256)]+
         [nn.Conv2d(128,output_channels,3,1,1)])
            
        #self.detours=nn.ModuleList(
        # [nn.Sequential(nn.Conv2d(64,2,7,1,3)),
        # nn.Sequential(nn.Conv2d(64,2,7,1,3))]+
        # [nn.Sequential(nn.Conv2d(128,2,7,1,3)) for i in range(n_128+1)]+
        # [nn.Sequential(nn.Conv2d(256,2,7,1,3)) for i in range(n_256+1)])
        

    def forward(self, x):
       
        for i, l in enumerate(self.convs):
            x = self.convs[i](x)
            #out.append(self.detours[i](x))
        
        return [x]












class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self,up_channels, skip_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(up_channels, up_channels, kernel_size=2, stride=2)

        self.conv = DoubleConv(up_channels+skip_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(512,512, 256)
        self.up2 = Up(256,256, 128)
        self.up3 = Up(128,128, 64)
        self.up4 = Up(64,64, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return [logits]
        
        
        
    #           self.down1 = Down(64, 128)
    #     self.down2 = Down(128, 256)
    #     self.down3 = Down(256, 256)
    #     self.up1 = Up(512, 128, 'bilinear')
    #     self.up2 = Up(256,64, 'bilinear')
    #     self.up3 = Up(128, 64, 'bilinear')
    #     self.outc = OutConv(64, n_classes)

   #   def forward(self, x):
    #     x1 = self.inc(x)
    #     x2 = self.down1(x1)
    #     x3 = self.down2(x2)
    #     x4 = self.down3(x3)

   #       x = self.up1(x4, x3)
    #     x = self.up2(x, x2)
    #     x = self.up3(x, x1)
    #     logits = self.outc(x)
    #     return [logits]


class Warp_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(Warp_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        
        self.xx,self.yy=torch.meshgrid(torch.arange(240),torch.arange(424))
        self.l=[0]
        self.in_conv=nn.Conv2d(n_channels,64,7,1,3)
        
        self.L=nn.Sequential(nn.Conv2d(n_channels,64,3,3,0,dilation=2),
        nn.LeakyReLU(),
        nn.BatchNorm2d(64),
        Flatten(),
        nn.Linear(79*140*64,128),
        nn.BatchNorm1d(128),
        nn.LeakyReLU(),
        nn.Linear(128,2*64),
        nn.Tanh())





  
        self.down1 = Down(128, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(512,512, 256)
        self.up2 = Up(256,256, 128)
        self.up3 = Up(128,128, 128)
        self.up4 = Up(128,128, 64)
        self.outc = OutConv(64, n_classes)
    def check(self):
        return self.l
        
    def forward(self, x):
        out=self.in_conv(x)
        b,c,h,w=out.shape
        offset=self.L(x).view(b*c,1,1,2)*20
        # if x[0,0,0,0]<1 and x[0,0,0,0]>0:
        #    print(offset.view(b,c,-1)[0])
   
        ind=torch.stack((self.yy,self.xx),dim=-1).cuda()
        ind=ind.repeat(b*c,1,1,1).cuda()
        grid=ind+offset
        grid=(2*grid/torch.tensor([w,h]).cuda().view(1,1,1,2))-1
        x1=nn.functional.leaky_relu(torch.cat((out,torch.nn.functional.grid_sample(out.view(b*c,1,h,w),grid,mode='bilinear',padding_mode='border',align_corners=False).view(b,c,h,w)),dim=1)  )
        self.l=offset.view(b,c,2)[0].detach()
     
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return [logits]
        
     

class Edge_generator(nn.Module):
    def __init__(self, n_channels, n_classes=1, bilinear=False):
        super(Edge_generator, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        
        self.down1 = Down(n_channels, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 64)
        
        self.noise_tf=nn.Sequential(
        nn.Linear(100,128),
        nn.BatchNorm1d(128),
        nn.Linear(128,2912),
        nn.BatchNorm1d(2912)
        )
        
        self.up1 = Up(32,64, 128, False)
        self.up2 = Up(128, 64, 128, False)
        self.up3 = Up(128, 32, 128, False)
        self.up4 = Up(128, n_channels, 128, False)
        
        self.outc = OutConv(128, n_classes)
    def forward(self, x):
        z=torch.randn(x.shape[0],100).cuda()
        x1 = x
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        z=torch.randn(x.shape[0],100).cuda()
        x5 = self.noise_tf(z).view(-1,32,7,13)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        logits = self.outc(self.up4(x, x1))
        return [torch.tanh(logits)]
    



class Discriminator(nn.Module):
    def __init__(self, in_channels=4):
        super(Discriminator, self).__init__()
        
        
       

        self.model = nn.Sequential(
            nn.Conv2d(in_channels,64,4,2,1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64,128,4,2,1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128,256,4,2,1),#30,53
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((16,32)),
            nn.Conv2d(256,256,4,2,1),#8,16
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256,512,3,(1,2),(1,1)),#8,8
            #nn.InstanceNorm2d(512),
            nn.LeakyReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(512,512,2,2),#2,2
            
            nn.LeakyReLU(),
            
            Flatten(),
            nn.Linear(2048,128),
            #nn.BatchNorm1d(128),
            Unsqueeze(),
            #nn.InstanceNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128,128),
           # nn.InstanceNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128,1)
        )

    def forward(self, img):
        # Concatenate image and condition image by channels to produce input
        return self.model(img)



# class Discriminator(nn.Module):
#     def __init__(self, in_channels=4):
#         super(Discriminator, self).__init__()
#         
#         
#         def discriminator_block(in_filters, out_filters, normalization=False):
#             """Returns downsampling layers of each discriminator block"""
#             layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
#             if normalization:
#                 layers.append(nn.InstanceNorm2d(out_filters))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             return layers
# 
#         self.model = nn.Sequential(
#             *discriminator_block(in_channels, 64, normalization=True),
#             *discriminator_block(64, 128),
#             *discriminator_block(128, 256),
#             *discriminator_block(256, 256),
#             *discriminator_block(256, 256),
#             Flatten(),
#             nn.Linear(4608,128),
#             Unsqueeze(),
#             nn.InstanceNorm1d(128),
#             nn.LeakyReLU(),
#             nn.Linear(128,64),
#             nn.InstanceNorm1d(64),
#             nn.LeakyReLU(),
#             nn.Linear(64,1)
#         )
# 
#     def forward(self, img):
#         # Concatenate image and condition image by channels to produce input
#         return self.model(img)


class Patch_Discriminator(nn.Module):
    def __init__(self, in_channels=4):
        super(Patch_Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=False):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=1, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 256),
            nn.Conv2d(256, 1, 4, bias=False)
        )

    def forward(self, img):
        # Concatenate image and condition image by channels to produce input
        return self.model(img)


class multi_Discriminator(nn.Module):
    def __init__(self, in_channels=4):
        super(multi_Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=False):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.l1 = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128))
        self.l2 = nn.Sequential(
            *discriminator_block(128, 256))
        self.l3 = nn.Sequential(
            *discriminator_block(256, 512))
        
        self.o1=nn.Conv2d(128, 1, 4, bias=False)
        self.o2=nn.Conv2d(256, 1, 4, bias=False)
        self.o3=nn.Conv2d(512, 1, 4, bias=False)
        

    def forward(self, img):
        x=self.l1(img)
        o1=self.o1(x)
        x=self.l2(x)
        o2=self.o2(x)
        x=self.l3(x)
        o3=self.o3(x)
        
        return o1,o2,o3



class D_Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = nn.Sequential(nn.Conv2d(in_channels,out_channels,3,1,1),nn.LeakyReLU())

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
    
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class D_Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels,3,1,1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Discriminator_mask(nn.Module):
    def __init__(self, in_channels=4):
        super(Discriminator_mask, self).__init__()

        self.inc = nn.Sequential(nn.Conv2d(in_channels, 128,3,1,1),nn.LeakyReLU())
        self.down1 = D_Down(128, 256)
        self.down2 = D_Down(256, 256)
        self.up1 = D_Up(512, 256)
        self.up2 = D_Up(384, 64)
        self.outc = nn.Sequential(nn.Conv2d(64,1,3,1,1))
        

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        logits = self.outc(x)
        return logits

def nparams(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp





       
        

    def forward(self, x):
        out=self.in_c(x)
        return out


class Features2flow(nn.Module):
    def __init__(self, in_channels=32):
        super(Features2flow, self).__init__()

        self.in_c = nn.Sequential(nn.Conv2d(in_channels, 4,kernel_size=1),
        # nn.BatchNorm2d(16),
        # nn.LeakyReLU(),
        # nn.Conv2d(16, 4,kernel_size=1),
        )
        
        

    def forward(self, x):
        out=self.in_c(x)
        return out



        
class Flow2features(nn.Module):
    def __init__(self, in_channels=4):
        super(Flow2features, self).__init__()

        self.in_c = nn.Sequential(nn.Conv2d(in_channels,32,1))
        
        self.res1=nn.Sequential(
        nn.Conv2d(32,32,3,1,1),
        nn.BatchNorm2d(32),
        nn.LeakyReLU())
        
        self.res2=nn.Sequential(
        nn.Conv2d(32,32,3,1,1),
        nn.BatchNorm2d(32),
        nn.LeakyReLU())

    def forward(self, x):
        out=self.in_c(x)
        out=out+self.res1(out)
        out=out+self.res2(out)
        return out



class Update(nn.Module):
    def __init__(self, in_channels=32*3):
        super(Update, self).__init__()
        self.pconv1=PartialConv2d(multi_channel='semi',return_mask=True,kernel_size=(5,5),padding=2,in_channels=in_channels,out_channels=64)
        self.bn1=nn.BatchNorm2d(in_channels)
        self.bn2=nn.BatchNorm2d(64)
        self.pconv2=PartialConv2d(multi_channel=False,return_mask=True,kernel_size=(3,3),padding=1,in_channels=64,out_channels=32)
      
    
    def forward(self,x,mask=None):
        x,new_mask=self.pconv1(self.bn1(x),mask)
        x,new_mask2=self.pconv2(F.leaky_relu(self.bn2(x)),new_mask)

        return x, new_mask2

class Res_Update(nn.Module):
    def __init__(self, in_channels=32*3):
        super(Res_Update, self).__init__()
        self.pconv1=PartialConv2d(multi_channel='semi',return_mask=True,kernel_size=(5,5),padding=2,in_channels=in_channels,out_channels=64)
        self.bn1=nn.BatchNorm2d(in_channels)
        self.bn2=nn.BatchNorm2d(64)
        self.pconv2=PartialConv2d(multi_channel=False,return_mask=True,kernel_size=(3,3),padding=1,in_channels=64,out_channels=64)
        self.bn3=nn.BatchNorm2d(64)
        self.pconv3=PartialConv2d(multi_channel=False,return_mask=True,kernel_size=(3,3),padding=1,in_channels=64,out_channels=32)
      
    
    def forward(self,x,mask=None):

        out1,new_mask=self.pconv1(self.bn1(x),mask)

        out2,_=self.pconv2(F.leaky_relu(self.bn2(out1)),new_mask)
        
        out3,_=self.pconv3(F.leaky_relu(self.bn3(out2+out1)),new_mask)
       
        return (x[:,32:64]*mask[:,32:33]+out3*new_mask*(1-mask[:,32:33]))/(mask[:,32:33]+new_mask*(1-mask[:,32:33])+10**-10), new_mask


class Res_Update2(nn.Module):
    def __init__(self, in_channels=32*3,update='pow'):
        super(Res_Update2, self).__init__()
        self.initial_mask=Initial_mask()
        self.pconv1=PartialConv2d(multi_channel='semi',return_mask=True,kernel_size=(3,3),padding=1,in_channels=in_channels,out_channels=64,update=update)
        self.bn1=nn.BatchNorm2d(in_channels)
        self.bn2=nn.BatchNorm2d(64)
        self.pconv2=PartialConv2d(multi_channel=False,return_mask=True,kernel_size=(3,3),padding=1,in_channels=64,out_channels=32,update=update)
        # self.bn3=nn.BatchNorm2d(64)
        # self.pconv3=PartialConv2d(multi_channel=False,return_mask=True,kernel_size=(3,3),padding=1,in_channels=64,out_channels=32)
      
    
    def forward(self,x,mask=None):

        out1,new_mask1=self.pconv1(F.leaky_relu(self.bn1(x)),mask)

        out2,new_mask2=self.pconv2(F.leaky_relu(self.bn2(out1)),new_mask1)
        
        # out3,_=self.pconv3(F.leaky_relu(self.bn3(out2+out1)),new_mask)
       
        return (x[:,32:64]+out2*new_mask2*(1-mask[:,32:33]))/(1+new_mask2*(1-mask[:,32:33])), new_mask2
    
    

class Res_Update3(nn.Module):
    def __init__(self, in_channels=32*3,update='pow'):
        super(Res_Update3, self).__init__()
        self.initial_mask=Initial_mask()
        self.pconv1=PartialConv2d(multi_channel='semi',return_mask=True,kernel_size=(3,3),padding=1,in_channels=in_channels,out_channels=64,update=update)
        self.bn1=nn.BatchNorm2d(in_channels)
        self.bn2=nn.BatchNorm2d(64)
        self.pconv2=PartialConv2d(multi_channel=False,return_mask=True,kernel_size=(3,3),padding=1,in_channels=64,out_channels=32,update=update)
        # self.bn3=nn.BatchNorm2d(64)
        # self.pconv3=PartialConv2d(multi_channel=False,return_mask=True,kernel_size=(3,3),padding=1,in_channels=64,out_channels=32)
      
    
    def forward(self,x,mask=None):

        out1,new_mask=self.pconv1(F.leaky_relu(self.bn1(x)),mask)

        out2,new_mask=self.pconv2(F.leaky_relu(self.bn2(out1)),new_mask)
        
        # out3,_=self.pconv3(F.leaky_relu(self.bn3(out2+out1)),new_mask)
       
        return (x[:,32:64]*mask[:,32:33]+out2*(1-mask[:,32:33])), new_mask


class Res_Update4(nn.Module):
    def __init__(self, in_channels=32*3,update='pow'):
        super(Res_Update4, self).__init__()
        self.pconv1=PartialConv2d(multi_channel='semi',return_mask=True,kernel_size=(3,3),padding=1,in_channels=in_channels,out_channels=64,update=update)
        self.bn1=nn.BatchNorm2d(in_channels)
        self.bn2=nn.BatchNorm2d(64)
        self.pconv2=PartialConv2d(multi_channel=False,return_mask=True,kernel_size=(3,3),padding=1,in_channels=64,out_channels=32,update=update)
        # self.bn3=nn.BatchNorm2d(64)
        # self.pconv3=PartialConv2d(multi_channel=False,return_mask=True,kernel_size=(3,3),padding=1,in_channels=64,out_channels=32)
      
    
    def forward(self,x,mask=None):

        out1,new_mask=self.pconv1(F.leaky_relu(self.bn1(x)),mask)

        out2,new_mask=self.pconv2(F.leaky_relu(self.bn2(out1)),new_mask)
        
        # out3,_=self.pconv3(F.leaky_relu(self.bn3(out2+out1)),new_mask)
       
        return (x[:,32:64]*mask[:,32:33]+out2*(1-mask[:,32:33])*new_mask), new_mask

class Initial_mask(nn.Module):
    def __init__(self):
        super(Initial_mask,self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(1,4,3,2,1,bias=False),nn.BatchNorm2d(4),nn.LeakyReLU())
        self.conv2=nn.Sequential(nn.Conv2d(4,8,3,2,1,bias=False),nn.BatchNorm2d(8),nn.LeakyReLU())
        self.conv3=nn.Sequential(nn.Conv2d(8,16,3,2,1,bias=False),nn.BatchNorm2d(16),nn.LeakyReLU())
        self.conv4=nn.Sequential(nn.Conv2d(16,16,3,2,1,bias=False),nn.BatchNorm2d(16),nn.LeakyReLU())
        
        self.up3=nn.Sequential(nn.Conv2d(32,16,1,bias=False),nn.BatchNorm2d(16),nn.LeakyReLU())
        self.up2=nn.Sequential(nn.Conv2d(24,8,1,bias=False),nn.BatchNorm2d(8),nn.LeakyReLU())
        self.up1=nn.Sequential(nn.Conv2d(12,8,1,bias=False),nn.BatchNorm2d(8),nn.LeakyReLU())
        self.out=nn.Conv2d(8,1,1,bias=False)
        
    
    def forward(self,x):
        _,_,h0,w0=x.shape
        x1=self.conv1(-x+.5)
        _,_,h1,w1=x1.shape
        x2=self.conv2(x1)
        _,_,h2,w2=x2.shape
        x3=self.conv3(x2)
        _,_,h3,w3=x3.shape
        x4=self.conv4(x3)
        
        y3=self.up3(torch.cat((F.interpolate(x4,size=(h3,w3),mode='bilinear',align_corners=True),x3),dim=1))
        y2=self.up2(torch.cat((F.interpolate(y3,size=(h2,w2),mode='bilinear',align_corners=True),x2),dim=1))
        y1=self.up1(torch.cat((F.interpolate(y2,size=(h1,w1),mode='bilinear',align_corners=True),x1),dim=1))
        out=self.out(F.interpolate(y1,size=(h0,w0),mode='bilinear',align_corners=True))
        return torch.sigmoid(out-1)*(x==0)+x*(x==1)

# class Res_Update4(nn.Module):
#     def __init__(self, in_channels=32*3,update='pow'):
#         super(Res_Update4, self).__init__()
#         
# 
#         
#         self.initial_mask=Initial_mask()
#         
#         self.t=torch.nn.Parameter(torch.tensor(1./100),requires_grad=True)
#         self.c=torch.nn.Parameter(torch.tensor(10./100),requires_grad=True)
#         
#         self.pconv1=PartialConv2d(multi_channel='semi',return_mask=True,kernel_size=(3,3),padding=1,in_channels=in_channels,out_channels=64,update=update)
#         self.bn1=nn.BatchNorm2d(in_channels)
#         self.bn2=nn.BatchNorm2d(64)
#         self.pconv2=PartialConv2d(multi_channel=False,return_mask=True,kernel_size=(3,3),padding=1,in_channels=64,out_channels=32,update=update)
#         # self.bn3=nn.BatchNorm2d(64)
#         # self.pconv3=PartialConv2d(multi_channel=False,return_mask=True,kernel_size=(3,3),padding=1,in_channels=64,out_channels=32)
#       
#     
#     def forward(self,x,mask=None):
# 
#         out1,new_mask1=self.pconv1(F.leaky_relu(self.bn1(x)),mask)
# 
#         out2,new_mask2=self.pconv2(F.leaky_relu(self.bn2(out1)),new_mask1)
#         
#         # out3,_=self.pconv3(F.leaky_relu(self.bn3(out2+out1)),new_mask)
#         
#        #torch.exp(self.t*100*mask[:,0:1])+torch.exp(self.t*100*(mask[:,32:33]+self.c*100))+torch.exp(self.t*100* mask[:,64:65])
#         #new_x=((torch.exp(torch.abs(self.t)*100*mask[:,0:1])-1)*x[:,:32]+x[:,32:64]*(torch.exp(torch.abs(self.t)*100*(mask[:,32:33]+torch.abs(self.c)*100))-1)+x[:,64:]*(torch.exp(torch.abs(self.t)*100* mask[:,64:65])-1))/(torch.exp(torch.abs(self.t)*100*mask[:,0:1])+torch.exp(torch.abs(self.t)*100*(mask[:,32:33]+torch.abs(self.c)*100))+torch.exp(torch.abs(self.t)*100* mask[:,64:65])-3+10**-10)
#         
#         return (x[:,32:64]*mask[:,32:33]+out2*new_mask2*(1-mask[:,32:33]))/(mask[:,32:33]+new_mask2*(1-mask[:,32:33])+10**5), new_mask2