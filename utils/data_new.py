import torch
import os
import random
import cv2
import matplotlib.pyplot as plt
import cvbase as cvb
import numpy as np
import torch.utils.data 
import utils.image as im
import utils.region_fill as rf
import pdb
from skimage.feature import canny



class dataset(torch.utils.data.Dataset):
    def __init__(self, config,val=False,test=False):
        super(dataset, self).__init__()
        self.n=config.n_frames
        self.config = config
        self.size = self.config.IMAGE_SHAPE
        self.size=config.IMAGE_SHAPE
        self.data_items = []
        if val:
            self.data_list = config.EVAL_LIST
        elif test:
            self.data_list = config.TEST_LIST
        else:
            self.data_list = config.TRAIN_LIST
            
        with open(self.data_list, 'r') as f:
            for line in f:
                flow_dir,boundary_dir,seg_dir,img_dir,mask_dir= None,None,None,None,None
                line = line.strip()
                line = line.strip(' ')
                line_split = line.split(' ')
                
                flow_dir = line_split[0:11]
                img_dir = line_split[11:22]
                if self.config.edge == 'grad' or self.config.edge == 'canny' or self.config.edge == None:
                    boundary_dir=line_split[33:44]
                elif self.config.edge == 'hed':
                    boundary_dir=line_split[44:55]
                if self.config.seg=='gt':
                    seg_dir = line_split[22:33]
            
            
                
                if self.config.get_mask:
                    mask_dir = line_split[11:22]# 11-22=raw Image
                    if not self.config.FIX_MASK:
                        mask_dir = [os.path.join(self.config.MASK_ROOT, x) for x in mask_dir]
                    else:
                        mask_dir = [os.path.join(self.config.MASK_ROOT) for x in mask_dir]
                else:
                    
                    self.data_items.append((flow_dir,boundary_dir,seg_dir,img_dir,mask_dir))


    def __len__(self):
        
        return len(self.data_items)

    def __getitem__(self, idx):
        flow_dir,boundary_dir,seg_dir,img_dir,mask_dir = self.data_items[idx]
       

        flow_set = []
        rflow_set=[]
        mask_set = []
        edge_set = []
        flow_masked_set = []
        rflow_masked_set=[]
        img_set=[]
        if self.config.MASK_MODE == 'bbox':
            tmp_bbox = im.random_bbox(self.config)
            tmp_mask = im.bbox2mask(self.config, tmp_bbox)
            tmp_mask = tmp_mask[0, 0, :, :]
            fix_mask = np.expand_dims(tmp_mask, axis=2)
        elif self.config.MASK_MODE == 'mid-bbox':
            tmp_mask = im.mid_bbox_mask(self.config)
            tmp_mask = tmp_mask[0, 0, :, :]
            fix_mask = np.expand_dims(tmp_mask, axis=2)
        



        for t,i  in enumerate(range(-self.n//2+6,6+self.n//2)): 
            tmp_flow = cvb.read_flow(flow_dir[i])
            tmp_flow = self._flow_tf(tmp_flow)
            
            if self.config.rflow :
                if flow_dir[i][-4:]=='.flo':
                    tmp_rflow=cvb.read_flow(flow_dir[i][:-3]+'rflo')
                else:
                    tmp_rflow=cvb.read_flow(flow_dir[i][:-4]+'flo')
                tmp_rflow = self._flow_tf(tmp_rflow)
                
            
            
            if self.config.edge=='grad':
                img=cv2.imread(boundary_dir[i])
                img=cv2.resize(img, (self.size[1], self.size[0]),interpolation=cv2.INTER_NEAREST)/ 127.5 - 1

                tmp_boundary = img[:,:,0:1]

            elif self.config.edge=='hed':
                img=cv2.imread(boundary_dir[i])
                img=cv2.resize(img, (self.size[1], self.size[0]),interpolation=cv2.INTER_NEAREST)/ 127.5 - 1
                tmp_boundary = img[:,:,0:1]#.reshape(self.size[0],self.size[1],1)
                
            elif self.config.edge=='canny':
                img=cvb.flow2rgb(tmp_flow)
                tmp_boundary = np.expand_dims(canny(img.mean(-1),1),-1)*2-1
                # plt.figure()
                # plt.imshow(tmp_boundary[:,:,0])
                # plt.show()
            
            elif self.config.edge=='none':
                #print(boundary_dir[i])
                img=cv2.imread(boundary_dir[i])
                img=cv2.resize(img, (self.size[1], self.size[0]))/ 127.5 - 1
                tmp_boundary = img[:,:,0:1]#.reshape(self.size[0],self.size[1],1)*0.
            elif self.config.seg=='gt':
                img=cv2.imread(seg_dir[i])
                img=cv2.resize(img, (self.size[1], self.size[0]),interpolation=cv2.INTER_NEAREST)/ 127.5 - 1
                tmp_boundary = img#[:,:,:].reshape(self.size[0],self.size[1],3)
            
            if self.config.image :
                path=img_dir[i]
                if path[-3:]=='png':
                    path=path[:-3]+'jpg'
                img=cv2.imread(path)
                
                tmp_img = self._img_tf(img)

           
            if self.config.get_mask:
                tmp_mask = cv2.imread(mask_dir[i],
                                    cv2.IMREAD_UNCHANGED)
                tmp_mask = self._mask_tf(tmp_mask)
            elif self.config.FIX_MASK:
                tmp_mask=fix_mask.copy()
            else:

                    tmp_mask = im.bbox2mask(self.config, tmp_bbox)
                    tmp_mask = tmp_mask[0, 0, :, :]
                    tmp_mask = np.expand_dims(tmp_mask, axis=2)
            
            if self.config.two_masks and t!=2 and t!=3:
                tmp_mask=tmp_mask*0.
            
            tmp_flow_masked = tmp_flow * (1. - tmp_mask)
            if self.config.rflow :
                tmp_rflow_masked = tmp_rflow * (1. - tmp_mask)

            if self.config.INITIAL_HOLE:
                img_dir=self.data_items[idx][3]
                tmp_flow_resized = cv2.resize(tmp_flow, (self.size[1] // 2, self.size[0] // 2))
                
                
                tmp_mask_resized = cv2.resize(tmp_mask, (self.size[1] // 2, self.size[0] // 2), cv2.INTER_NEAREST)
                
                
                tmp_flow_masked_small = tmp_flow_resized
                tmp_flow_masked_small[:, :, 0] = rf.regionfill(tmp_flow_resized[:, :, 0], tmp_mask_resized)
                tmp_flow_masked_small[:, :, 1] = rf.regionfill(tmp_flow_resized[:, :, 1], tmp_mask_resized)
                tmp_flow_masked = tmp_flow_masked + \
                                  tmp_mask * cv2.resize(tmp_flow_masked_small, (self.size[1], self.size[0]))

                
                if self.config.rflow :
                    tmp_rflow_resized = cv2.resize(tmp_rflow, (self.size[1] // 2, self.size[0] // 2))
                    tmp_rflow_masked_small = tmp_rflow_resized
                    tmp_rflow_masked_small[:, :, 0] = rf.regionfill(tmp_rflow_resized[:, :, 0], tmp_mask_resized)
                    tmp_rflow_masked_small[:, :, 1] = rf.regionfill(tmp_rflow_resized[:, :, 1], tmp_mask_resized)
                    tmp_rflow_masked = tmp_rflow_masked + \
                                    tmp_mask * cv2.resize(tmp_rflow_masked_small, (self.size[1], self.size[0]))

            flow_masked_set.append(tmp_flow_masked)
            flow_set.append(tmp_flow)
            mask_set.append(np.concatenate((tmp_mask,tmp_mask), axis=2))

            
            if self.config.edge is not None  or self.config.seg is not None:
                edge_set.append(tmp_boundary)
            if self.config.rflow:
                rflow_masked_set.append(tmp_rflow_masked)
                rflow_set.append(tmp_rflow)
            if self.config.image:
                img_set.append(tmp_img)
            
            
            

        flow_masked = np.stack(flow_masked_set, axis=2)
        gt_flow = np.stack(flow_set, axis=2)
        mask = np.stack(mask_set, axis=2)
    
        flow_masked = torch.from_numpy(flow_masked).permute(2,3,0,1).contiguous().float()
        gt_flow = torch.from_numpy(gt_flow).permute(2,3,0,1).contiguous().float()
        mask = torch.from_numpy(mask).permute(2,3,0,1).contiguous().float()
        
        if self.config.rflow:
            rflow_masked = torch.from_numpy(np.stack(rflow_masked_set, axis=2)).permute(2,3,0,1).contiguous().float()
            gt_rflow = torch.from_numpy(np.stack(rflow_set, axis=2)).permute(2,3,0,1).contiguous().float()
        else:
            rflow_masked=0
            gt_rflow=0
        
        if self.config.image :
            img = torch.from_numpy(np.stack(img_set, axis=2)).permute(2,3,0,1).contiguous().float()
        else:
            img=0
        
        if self.config.edge is not None or self.config.seg is not None:
            edge = torch.from_numpy(np.stack(edge_set, axis=2)).permute(2,3,0,1).contiguous().float()
        else:
            edge=0
   
            
    
        return flow_masked,rflow_masked, mask, gt_flow, gt_rflow, edge, img, tmp_bbox

    def _img_tf(self, img):
        img = cv2.resize(img, (self.size[1], self.size[0]))
        img = img / 127.5 - 1

        return img

    def _mask_tf(self, mask):
        mask = cv2.resize(mask, (self.size[1], self.size[0]),
                          interpolation=cv2.INTER_NEAREST)
        if self.config.enlarge_mask:
            enlarge_kernel = np.ones((self.config.enlarge_kernel, self.config.enlarge_kernel),
                                     np.uint8)
            tmp_mask = cv2.dilate(mask[:, :, 0], enlarge_kernel, iterations=1)
            mask[(tmp_mask > 0), :] = 255

        mask = mask[:,:,0]
        mask = np.expand_dims(mask, axis=2)
        mask = mask / 255
        return mask

    def _flow_tf(self, flow):
        origin_shape = flow.shape
        flow = cv2.resize(flow, (self.size[1], self.size[0]))
        flow[:, :, 0] = flow[:, :, 0].clip(-1. * origin_shape[1], origin_shape[1]) / origin_shape[1] * self.size[1]
        flow[:, :, 1] = flow[:, :, 1].clip(-1. * origin_shape[0], origin_shape[0]) / origin_shape[0] * self.size[0]

        return flow


