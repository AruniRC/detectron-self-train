
"""
Convert detections into format for IJBC-style evaluation on CS6.

By default the "val" split is considered, using validation videos listed in 
data/CS6/list_video_val.txt. 
The properly formatted detection text files are saved in a subfolder under 
the original detections' folder (DET_DIR). For basic usage, only this <DET_DIR> 
needs to be provided as input. 

Usage:
srun --pty --mem 50000 python tools/face/make_cs6_det_eval.py --split val \
                        --det_dir <DET_DIR>

Output files:
    
    <DET_DIR>/
        eval-dets_<split>/
            <video-name>.txt



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


DET_NAME = 'frcnn-R-50-C4-1x'
# DET_DIR = 'Outputs/evaluations/frcnn-R-50-C4-1x/cs6/sample-baseline-video/'
DET_DIR = 'Outputs/evaluations/frcnn-R-50-C4-1x/cs6/sample-baseline-video_conf-0.25/'
VIDEO_LIST_FILE = 'data/CS6/list_video_%s.txt'
SPLIT = 'val'
GT_ANNOT_DIR =  'data/CS6_annot'


def parse_args():
    parser = argparse.ArgumentParser(description='Creating CS6 ground truth data')
    parser.add_argument(
        '--exp_name',
        help='detector name', 
        default=DET_NAME
    )
    parser.add_argument(
        '--det_dir', 
        help='Folder containing raw detection files', 
        default=DET_DIR
    )
    parser.add_argument(
        '--split', help='Split (train, val, test)', 
        default=SPLIT
    )
    parser.add_argument(
        '--video_list', help='Path to CS6 videos listed in split', 
        default=VIDEO_LIST_FILE
    )
    parser.add_argument(
        '--gt_dir', help='Path to CS6 ground-truth annotations', 
        default=GT_ANNOT_DIR
    )
    return parser.parse_args()




if __name__ == '__main__':

    args = parse_args()
    args.video_list = args.video_list % args.split
    np.random.seed(0)

    out_dir = osp.join(args.det_dir, 'eval-dets_' + args.split)
    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    vid_list = np.loadtxt(args.video_list, dtype=str)
    if vid_list.size == 1:
        vid_list = [vid_list.tolist()] # work-around: iter over singleton np.array 

    for video_name in vid_list:
        print(video_name)
        # Groundtruth file and image list
        # NOTE: the detections must be written in the same order as images in GT file
        gt_file = osp.join(args.gt_dir, 'video_annots', 
                           video_name.split('.')[0] + '.txt')
        with open(gt_file) as f:
            im_list = [x.rstrip() for x in f if '.jpg' in x]
        im_list = np.array(im_list, dtype=str)

        # Detection results
        det_file = osp.join(args.det_dir, video_name.split('.')[0] + '.txt')
        if not osp.exists(det_file):
            print('Skipping video %s (detections file not found)' % video_name)
            continue
        det_dict = face_utils.parse_wider_gt(det_file)
        det_out_file = osp.join(out_dir, video_name.split('.')[0] + '.txt')
        
        with open(det_out_file, 'w') as fid_det:
            # Output bboxes lists for evaluation
            for i, im_name in enumerate(im_list):
                # E.g. 'frames/1100/1100_00002816.jpg' --> '1100_00002816'
                im_query = osp.splitext(osp.basename(im_name))[0]
                try:
                    dets = np.array( det_dict[im_query] )
                except KeyError:
                    dets = np.empty((0,5))
                fid_det.write(im_name + '\n')
                fid_det.write(str(dets.shape[0]) + '\n')
                for j in xrange(dets.shape[0]):
                    fid_det.write('%f %f %f %f %f\n' % ( dets[j, 0], dets[j, 1], 
                                                        dets[j, 2], dets[j, 3], 
                                                        dets[j, 4]) )
                if ((i + 1) % 100) == 0:
                    sys.stdout.write('. ')
                    sys.stdout.flush()

            print('\n')
        
        
        
