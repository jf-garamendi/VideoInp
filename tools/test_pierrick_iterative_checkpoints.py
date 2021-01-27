import argparse
import os
import glob
import video_completion
from tqdm import tqdm
import re

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--chk_root_path', help='root folder where the checkpoints (each in a different folder inside root) are saved')

    parser.add_argument('--flow2features_weights', default=None,
                        help='Path to the weights of the  encoder (flow to features) network architecture')
    parser.add_argument('--features2flow_weights', default=None,
                        help='Path to the weights of the  decoder (features to flow) network architecture')
    parser.add_argument('--update_weights', default=None, help='Path to the weights of the update network architecture')
    parser.add_argument('--update_number_model', default='2', help='Class number of the update network architecture')
    parser.add_argument('--kind_update', default='pow', help='Class number of the update network architecture')

    parser.add_argument('--mode', default='object_removal', help="modes: object_removal / video_extrapolation")
    parser.add_argument('--seamless', action='store_true', help='Whether operate in the gradient domain')
    parser.add_argument('--video_path', default='../data/tennis', help="dataset for evaluation")
    parser.add_argument('--path_mask', default='../data/tennis_mask', help="mask for object removal")
    parser.add_argument('--outroot', default='../result/', help="output directory")

    parser.add_argument('--verbose', action='store_true', help='use small model')
    parser.add_argument('--verbose_path', default='../verbose_output', help='where intermediate results will be saved')

    # RAFT
    parser.add_argument('--opticalFlow_model', default='../weight/raft-things.pth',
                        help="restore checkpoint for computing OF")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    args = parser.parse_args()

    initial_verbose_path = args.verbose_path

    #Read the root folder
    for chk_name in os.listdir(args.chk_root_path):
        print('*** Folder: ' + chk_name +' ***')


        n = 0
        for s in os.listdir(os.path.join(args.chk_root_path, chk_name, "ckpt")):
            if 'update' in s:
                n = max(n, int(s.strip("update_").strip(".pth")))

        if n == 0:
            print('no save')
        else:
            if 'pow' in chk_name:
                args.kind_update = 'pow'
            elif 'pol' in chk_name:
                args.kind_update = 'pol'

            if '2' in chk_name:
                args.update_number_model = '2'
            elif '3' in chk_name:
                args.update_number_model = '3'
            elif '4' in chk_name:
                args.update_number_model = '4'

            args.flow2features_weights = os.path.join(args.chk_root_path, chk_name,"ckpt","flow_2F_"+str(n)+".pth")
            args.features2flow_weights = os.path.join(args.chk_root_path, chk_name,"ckpt","F2flow_"+str(n)+".pth")
            args.update_weights = os.path.join(args.chk_root_path, chk_name,"ckpt","update_"+str(n)+".pth")

            args.verbose_path = os.path.join(initial_verbose_path, chk_name)
            args.outroot = os.path.join(args.outroot, chk_name)

            args.seamless = True
            args.verbose = True

            try:
                video_completion.main(args)
            except:
                print('SOME ERROR IN ' + chk_name)

    #each folder inside the root corresponds to a checkpoint. Run it
