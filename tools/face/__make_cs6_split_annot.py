
"""
Create ground-truth annotation files for IJBC-style evaluation on CS6.

By default the "val" split is considered, using validation videos listed in 
data/CS6/list_video_val.txt. 

NOTE: create symlink to "/mnt/nfs/work1/elm/arunirc/Data/CS6_annots/" at "data/CS6_annots". 

Usage:
srun --pty --mem 50000 python tools/face/make_cs6_split_annot.py --split val

Output files:
data/CS6_annots
    cs6_annot_eval_imlist_val.txt
    cs6_annot_eval_val.txt

"""

from __future__ import absolute_import
from __future__ import division

import matplotlib 
matplotlib.use('Agg') 

import sys
sys.path.append('./tools')
import _init_paths
import numpy as np
import os
import argparse
import os.path as osp
import time
from six.moves import xrange
import utils.face_utils as face_utils


GT_VIDEO_LIST = 'data/CS6/list_video_%s.txt'
GT_ANNOT_DIR =  '/mnt/nfs/work1/elm/arunirc/Data/CS6_annots/video_annots'
DET_DIR = '/mnt/nfs/work1/elm/arunirc/Research/face-faster-rcnn-ohem/output/CS6/dets-resnet101-baseline-frames'
NUM_IM_VID = 20  # number of images to be sampled per video (for subset creation)


DEBUG = False

def parse_args():
    parser = argparse.ArgumentParser(description='Creating CS6 ground truth data')
    parser.add_argument(
        '--gt_dir', help='Path to CS6 ground-truth', 
        default=GT_ANNOT_DIR
    )
    parser.add_argument(
        '--split', help='Split (train, val, test)', 
        default='val'
    )
    parser.add_argument(
        '--video_list', help='Path to CS6 videos listed in split', 
        default=GT_VIDEO_LIST
    )
    # parser.add_argument(
    #     '--subset', help='Create a subset for quick eval', action='store_true',
    #     default=False
    # )
    return parser.parse_args()




if __name__ == '__main__':
    

    args = parse_args()
    args.video_list = args.video_list % args.split
    np.random.seed(0) 

    # -----------------------------------------------------------------------------------
    # Data setup
    # -----------------------------------------------------------------------------------

    # Ground truth
    vid_list = np.loadtxt(args.video_list, dtype=str)


    # Outputs
    gt_out_dir = osp.dirname(args.gt_dir)
    gt_out_file = osp.join(gt_out_dir, 'cs6_annot_eval_%s.txt' % args.split)

    gt_imlist_file = osp.join(gt_out_dir, 'cs6_annot_eval_imlist_%s.txt' % args.split)

    
    # -----------------------------------------------------------------------------------
    # Eval-format ground-truth annots for CS6
    # -----------------------------------------------------------------------------------
    with open(gt_out_file, 'w') as fid_gt:
        with open(gt_imlist_file, 'w') as fid_imlist:

            for video_name in vid_list:

                # Load ground-truth annots for that video
                gt_file = osp.join(args.gt_dir, 
                                   video_name.split('.')[0] + '.txt')
                gt_annots = face_utils.parse_wider_gt(gt_file)
                if len(gt_annots) == 0:
                    continue # no gt faces in this video
                image_list = np.array( list(gt_annots.keys()) )

                # # Select a subset of frames, or use all frames (much slower)
                # if args.subset:
                #     assert len(image_list) != 0
                #     subset_size = min( (NUM_IM_VID, len(image_list)) )
                #     sel = np.random.randint(len(image_list), size=NUM_IM_VID)
                #     image_list = image_list[sel]

                print('Video annot: %s' % gt_file)

                # Output bboxes lists for evaluation
                for i, im_name in enumerate(image_list):

                    # Writing to ground-truth text file
                    annot = np.array( gt_annots[im_name] )
                    fid_gt.write(im_name + '\n')
                    fid_gt.write(str(annot.shape[0]) + '\n')
                    for j in xrange(annot.shape[0]):
                        fid_gt.write('%f %f %f %f\n' % ( annot[j, 0], annot[j, 1], 
                                                         annot[j, 2], annot[j, 3]) )

                    # Writing image names (order of images must match for imlist and annots)
                    fid_imlist.write(im_name + '\n')

                    if ((i + 1) % 100) == 0:
                        sys.stdout.write('. ')
                        sys.stdout.flush()

                print('\n')
        
        
        
