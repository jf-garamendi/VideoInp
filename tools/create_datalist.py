import os
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()


    parser.add_argument('--dataset_root', type=str, default="/home/pierrick/Documents/GitHub/data/Youtube_VOS_2018/train/JPEGImages")
    parser.add_argument('--seg_root', type=str, default="/home/pierrick/Documents/GitHub/data/Youtube_VOS_2018/train/Annotations")
    parser.add_argument('--boundary_root', type=str, default="/home/pierrick/Documents/GitHub/data/Youtube_VOS_2018/train/Boundaries")
    parser.add_argument('--hed_root', type=str, default="/home/pierrick/Documents/GitHub/data/Youtube_VOS_2018/train/HED")
    parser.add_argument('--output_txt_path', type=str, default="None")
    

    args = parser.parse_args()
    return args

def gen_flow_initial_list(root,seg_root,boundary_root,hed_root,output_txt_path,f=""):

    flow_root=os.path.join(root,"Flow")

    output_txt = open(output_txt_path, 'a')
    flow_list = [x for x in os.listdir(flow_root) if '.flo' in x]
    flow_no_list = sorted([int(x[:5]) for x in flow_list])
    flow_start_no = 0
    
    flow_num = len(flow_list)
    #print(flow_no_list)
    if flow_num > 11:
        video_num = 0
    
        for i in range(flow_start_no , flow_start_no + flow_num - 10):
            if True:
                
                for k in range(11):
                    flow_no = np.clip(i+k, a_min=flow_start_no, a_max=len(flow_list)-1)
                    #print(i+k,flow_no,len(flow_list))
                    
                    output_txt.write(flow_root+'/%05d.flo' % (flow_no_list[flow_no])) #5 for VOS
                    output_txt.write(' ')
        
                for k in range(11):
                    flow_no = np.clip(i + k, a_min=flow_start_no, a_max=len(flow_list)-1)
                    output_txt.write(root+'/%05d.png' % (flow_no_list[flow_no]))
                    output_txt.write(' ')
                    
                for k in range(11):
                    flow_no = np.clip(i + k, a_min=flow_start_no, a_max=len(flow_list)-1)
                    output_txt.write(seg_root+'/%05d.png' % (flow_no_list[flow_no]))
                    output_txt.write(' ')
                
                for k in range(11):
                    flow_no = np.clip(i + k, a_min=flow_start_no, a_max=len(flow_list)-1)
                    output_txt.write(boundary_root+'/%05d.png' % (flow_no_list[flow_no]))
                    output_txt.write(' ')
                
                for k in range(11):
                    flow_no = np.clip(i + k, a_min=flow_start_no, a_max=len(flow_list)-1)
                    output_txt.write(hed_root+'/%05d.jpg' % (flow_no_list[flow_no]))
                    output_txt.write(' ')
        
                # output_txt.write(flow_root+'/%05d.flo' % (i+5))
                # output_txt.write(' ')
                output_txt.write(str(video_num))
                output_txt.write('\n')
        rflow_list = [x for x in os.listdir(flow_root) if '.rflo' in x]
        rflow_no_list = sorted([int(x[:5]) for x in rflow_list])
        rflow_start_no = 0
        
        rflow_num = len(rflow_list)
        for i in range(flow_start_no- 0, flow_start_no + rflow_num - 10):
            if True:
                for k in range(11):
                    rflow_no = np.clip(i+k, a_min=rflow_start_no, a_max=len(rflow_list)-1)
                    output_txt.write(flow_root+'/%05d.rflo' % (rflow_no_list[rflow_no]))
                    output_txt.write(' ')
        
                for k in range(11):
                    rflow_no = np.clip(i + k, a_min=rflow_start_no, a_max=len(rflow_list)-1)
                    output_txt.write(root+'/%05d.png' % (rflow_no_list[rflow_no]))
                    output_txt.write(' ')
                for k in range(11):
                    rflow_no = np.clip(i + k, a_min=rflow_start_no, a_max=len(rflow_list)-1)
                    output_txt.write(seg_root+'/%05d.png' % (rflow_no_list[rflow_no]))
                    output_txt.write(' ')
                
                for k in range(11):
                    rflow_no = np.clip(i + k, a_min=rflow_start_no, a_max=len(rflow_list)-1)
                    output_txt.write(boundary_root+'/%05d.png' % (rflow_no_list[rflow_no]))
                    output_txt.write(' ')
                
                
                for k in range(11):
                    rflow_no = np.clip(i + k, a_min=rflow_start_no, a_max=len(rflow_list)-1)
                    output_txt.write(hed_root+'/%05d.jpg' % (rflow_no_list[rflow_no]))
                    output_txt.write(' ')
        
                # output_txt.write('%05d.rflo' % (i+5))  #TO BE REMOVED ACCORDING TO TRAINIING ROCEDURE (SEE GIT)
                # output_txt.write(' ')
                output_txt.write(str(video_num))
                output_txt.write('\n')
    
        output_txt.close()



def gen_flow_refine_test_mask_list(flow_root, output_txt_path):
    output_txt = open(output_txt_path, 'w')

    flow_list = [x for x in os.listdir(flow_root) if 'flo' in x]
    flow_no_list = [int(x[:5]) for x in flow_list]
    flow_start_no = min(flow_no_list)
    flow_num = len(flow_list) // 2

    assert flow_num > 11

    for i in range(flow_start_no - 5, flow_start_no + flow_num - 4):
        gt_flow_no = [0, 0]
        f_flow_no = []
        for k in range(11):
            flow_no = np.clip(i + k, a_min=flow_start_no, a_max=flow_start_no + flow_num - 1)
            f_flow_no.append(int(flow_no))
            output_txt.write('%05d.flo' % flow_no)
            if k == 5:
                gt_flow_no[0] = flow_no
            output_txt.write(' ')

        r_flow_no = []
        for k in range(11):
            flow_no = np.clip(i + k, a_min=flow_start_no, a_max=flow_start_no + flow_num - 1)
            r_flow_no.append(int(flow_no))
            if k == 5:
                gt_flow_no[1] = flow_no
            output_txt.write('%05d.rflo' % flow_no)
            output_txt.write(' ')

        for k in range(11):
            output_txt.write('%05d.png' % f_flow_no[k])
            output_txt.write(' ')
        for k in range(11):
            output_txt.write('%05d.png' % r_flow_no[k])
            output_txt.write(' ')


        output_path = ','.join(['%05d.flo' % gt_flow_no[0],
                                '%05d.rflo' % gt_flow_no[1]])
        output_txt.write(output_path)
        output_txt.write(' ')
        output_txt.write(str(0))
        output_txt.write('\n')

    output_txt.close()
    
    
    
def main():
    root="/home/pierrick/Documents/GitHub/data/Youtube_VOS_2018/train/Annotations/"
    l=os.listdir(root)
    blacklist=[]
    L=len(next(os.walk(root))[1])
    for dir in l:
        a=[]
        for f in os.listdir(os.path.join(root,dir)):
            if f[-4:]==".png":
                a.append(int(f.strip(".png")))
        a=sorted(a)
        for i in range(len(a)-1):
            if a[i]+5!=a[i+1]:
                #print(dir,a[i]+5)
                blacklist.append(dir)
    args=parse_args()
    #os.chdir("/home/pierrick/Documents/GitHub/Deep-Flow-Guided-Video-Inpainting")
    open(args.output_txt_path, 'w').close()
    for i,f in enumerate(os.listdir(args.dataset_root)):
        if f  in blacklist:
            print(f,"blacklisted")
        elif 'train' in args.output_txt_path:
            if i/L<.9:
                frame_dir=os.path.join(args.dataset_root,f)
                seg_root=os.path.join(args.seg_root,f)
                boundary_root=os.path.join(args.boundary_root,f)
                hed_root=os.path.join(args.hed_root,f)
                gen_flow_initial_list(frame_dir,seg_root,boundary_root,hed_root,args.output_txt_path,f)
        elif 'val' in args.output_txt_path:
            if .9<=i/L<.95:
                frame_dir=os.path.join(args.dataset_root,f)
                seg_root=os.path.join(args.seg_root,f)
                boundary_root=os.path.join(args.boundary_root,f)
                hed_root=os.path.join(args.hed_root,f)
                gen_flow_initial_list(frame_dir,seg_root,boundary_root,hed_root,args.output_txt_path,f)
        elif 'test' in args.output_txt_path:
            if .95<=i/L:
                frame_dir=os.path.join(args.dataset_root,f)
                seg_root=os.path.join(args.seg_root,f)
                boundary_root=os.path.join(args.boundary_root,f)
                hed_root=os.path.join(args.hed_root,f)
                gen_flow_initial_list(frame_dir,seg_root,boundary_root,hed_root,args.output_txt_path,f)
             

if __name__ == '__main__':
    main()