import os
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--txt_file', type=str,default="/home/pierrick/Documents/GitHub/data/DAVIS-2017-trainval-480p/DAVIS/ImageSets/2017/train.txt")
    parser.add_argument('--dataset_root', type=str, default="/home/pierrick/Documents/GitHub/data/DAVIS-2017-trainval-480p/DAVIS/JPEGImages/480p/")
    parser.add_argument('--seg_root', type=str, default="/home/pierrick/Documents/GitHub/data/DAVIS-2017-trainval-480p/DAVIS/Annotations/480p/")
    parser.add_argument('--boundary_root', type=str, default="/home/pierrick/Documents/GitHub/data/DAVIS-2017-trainval-480p/DAVIS/Boundaries/")
    parser.add_argument('--out', type=str, default="None")
    parser.add_argument('--hed_root', type=str, default="/home/pierrick/Documents/GitHub/data/DAVIS-2017-trainval-480p/DAVIS/HED")
    

    args = parser.parse_args()
    return args

def gen_flow_initial_list(root,seg_root,boundary_root,hed_root, out):
    flow_root=os.path.join(root,"Flow")
    output_txt = open(out, 'a')
    flow_list = [x for x in os.listdir(flow_root) if 'flo' in x]
    flow_no_list = [int(x[:5]) for x in flow_list]
    flow_start_no = min(flow_no_list)
    flow_num = len(flow_list) // 2

    assert flow_num > 11
    video_num = 0

    for i in range(flow_start_no - 5, flow_start_no + flow_num - 5):
        if i>0 and i+12<flow_num:
            for k in range(11):
                flow_no = np.clip(i+k, a_min=flow_start_no, a_max=flow_start_no + flow_num - 1)
                output_txt.write(flow_root+'/%05d.flo' % flow_no)
                output_txt.write(' ')
    
            for k in range(11):
                flow_no = np.clip(i + k, a_min=flow_start_no, a_max=flow_start_no + flow_num - 1)
                output_txt.write(root+'/%05d.png' % flow_no)
                output_txt.write(' ')
            
            for k in range(11):
                flow_no = np.clip(i + k, a_min=flow_start_no, a_max=flow_start_no + flow_num - 1)
                output_txt.write(seg_root+'/%05d.png' % flow_no)
                output_txt.write(' ')
                
            for k in range(11):
                flow_no = np.clip(i + k, a_min=flow_start_no, a_max=flow_start_no + flow_num - 1)
                output_txt.write(boundary_root+'/%05d.png' % flow_no)
                output_txt.write(' ')
            
            
            for k in range(11):
                flow_no = np.clip(i + k, a_min=flow_start_no, a_max=flow_start_no + flow_num - 1)
                output_txt.write(hed_root+'/%05d.jpg' % (flow_no))
                output_txt.write(' ')
        
                
            # output_txt.write(flow_root+'/%05d.flo' % (i+5))
            # output_txt.write(' ')
            output_txt.write(str(video_num))
            output_txt.write('\n')

    for i in range(flow_start_no - 5, flow_start_no + flow_num - 4):
        if i>0 and i+12<flow_num:
            for k in range(11):
                flow_no = np.clip(i+k, a_min=flow_start_no, a_max=flow_start_no + flow_num)
                output_txt.write(flow_root+'/%05d.rflo' % flow_no)
                output_txt.write(' ')
    
            for k in range(11):
                flow_no = np.clip(i + k, a_min=flow_start_no, a_max=flow_start_no + flow_num)
                output_txt.write(root+'/%05d.png' % flow_no)
                output_txt.write(' ')
            
            for k in range(11):
                flow_no = np.clip(i + k, a_min=flow_start_no, a_max=flow_start_no + flow_num - 1)
                output_txt.write(seg_root+'/%05d.png' % flow_no)
                output_txt.write(' ')
            
            for k in range(11):
                flow_no = np.clip(i + k, a_min=flow_start_no, a_max=flow_start_no + flow_num - 1)
                output_txt.write(boundary_root+'/%05d.png' % flow_no)
                output_txt.write(' ')
            
            
            for k in range(11):
                flow_no = np.clip(i + k, a_min=flow_start_no, a_max=flow_start_no + flow_num - 1)
                output_txt.write(hed_root+'/%05d.jpg' % (flow_no))
                output_txt.write(' ')
    
    
            # output_txt.write('%05d.rflo' % (i+5))  #TO BE REMOVED ACCORDING TO TRAINIING ROCEDURE (SEE GIT)
            # output_txt.write(' ')
            output_txt.write(str(video_num))
            output_txt.write('\n')

    output_txt.close()



def gen_flow_refine_test_mask_list(flow_root, out):
    output_txt = open(out, 'w')

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
            flow_no = np.clip(i + k, a_min=flow_start_no, a_max=flow_start_no + flow_num)
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
    args=parse_args()

    with open(args.txt_file) as f:
        lines = f.read().splitlines()
    open(args.out, 'w').close()
    for f in os.listdir(args.dataset_root):
        if f in lines:
            frame_dir=os.path.join(args.dataset_root,f)
            seg_root=os.path.join(args.seg_root,f)
            boundary_root=os.path.join(args.boundary_root,f)
            hed_root=os.path.join(args.hed_root,f)
            gen_flow_initial_list(frame_dir,seg_root,boundary_root,hed_root,args.out)

if __name__ == '__main__':
    main()