#!/usr/bin/env python2


"""
Convert the "CS6 format" video detections to the mining/tracker format. 
A symlink 'data/CS6' should point to the CS6 data root location. 

VIDEO DET FORMAT:
<video-name>_<0-starting-frame-number>
<num-dets>
[x1, y1, w, h, score]
[x1, y1, w, h, score]
... 

MINING DET FORMAT:
<frame-num>  # starting from 1
<num-dets>
[x1, y1, w, h, score]
[x1, y1, w, h, score]
... 


DET_FOLDER = Outputs/evaluations/<detector-name>/cs6/sample-baseline-video
OUT_FOLDER = ../DET_FOLDER/mining-detections/<split>_<conf-threshold>

FILES:
OUT_FOLDER
- <vid-name>.txt
- ...


TODO - Usage (on slurm cluster):

srun --pty --mem 50000 --gres gpu:1 python tools/face/convert_dets_mining_format.py ...

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import distutils.util
import os
import sys
import pprint
import subprocess
from collections import defaultdict
from six.moves import xrange
import os.path as osp
import time

# Use a non-interactive backend
import matplotlib
matplotlib.use('Agg')

import numpy as np
import cv2
import sys
sys.path.append('./tools')

import _init_paths
import utils.face_utils as face_utils



DET_NAME = 'frcnn-R-50-C4-1x'
DET_DIR = 'Outputs/evaluations/frcnn-R-50-C4-1x/cs6/sample-baseline-video/'
VIDEO_LIST_FILE = 'list_video_val.txt'  # parent folder is 'data/CS6'
CONF_THRESH_LIST = '0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8'
SPLIT = 'val'

# OUT_DIR = 'Outputs/evaluations/%s/cs6/mining-detections'  # usually unchanged


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert the "CS6 format" detections to the mining format.'
    )
    parser.add_argument(
        '--exp_name',
        help='detector name', 
        default=DET_NAME
    )
    parser.add_argument(
        '--output_dir',
        help='Saves detection txt files in mining-algo format',
        default='',
        type=str
    )
    parser.add_argument(
        '--thresh_list',
        help='Comma-separated list of thresholds on class score',
        default=CONF_THRESH_LIST,
        type=str
    )
    parser.add_argument(
        '--det_dir', 
        help='Folder containing raw detection files', 
        default=DET_DIR
    )
    parser.add_argument(
        '--split', 
        help="'train' or 'val' (default 'val')", 
        default=SPLIT
    )
    parser.add_argument(
        '--video_list_file', 
        help='Path to file listing the video-dets to process', 
        default=VIDEO_LIST_FILE
    )
    return parser.parse_args()



def write_formatted_dets(video_name, out_file_name, det_dict, conf_thresh):
    '''
    Write the mining-formatted detections to a text file.

    Output dets format: 
    <frame-num>  # starting from 1
    <num-dets>
    [x1, y1, w, h, score]
    [x1, y1, w, h, score]

    Stops writing frames to text file after the last frame in `det_dict`.

    '''
    frame_names = list(det_dict.keys())
    last_frame_name = sorted(frame_names)[-1]
    last_frame = int(last_frame_name.split('_')[-1])

    with open(out_file_name, 'w') as fid:
        #  frames (1,...,N+1) for mining-format
        for i in xrange(1, last_frame+1):
            frame = '%s_%08d' % (video_name, i-1) # det_dict is zero-starting
            try:
                dets = np.array(det_dict[frame])
                keep = np.where(dets[:, 4] > conf_thresh)
                dets = dets[keep]
            except KeyError:
                dets = np.empty((0,5))

            # Writing to text file
            fid.write('%d\n' % i)
            fid.write(str(dets.shape[0]) + '\n')
            for j in xrange(dets.shape[0]):
                fid.write('%f %f %f %f %f\n' % ( dets[j, 0], dets[j, 1], 
                                                 dets[j, 2], dets[j, 3], 
                                                 dets[j, 4]) )




if __name__ == '__main__':

    args = parse_args()

    if not args.output_dir:
        args.output_dir = osp.abspath(
                            osp.join(args.det_dir, '..', 'mining-detections' ))
    print('Called with args:')
    print(args)

    # List of video files
    with open( osp.join('data/CS6', args.video_list_file), 'r' ) as f:
        video_list = [ y.split('.')[0] for y in [ x.strip() for x in f ] ]    

    # List of detector conf. thresholds
    thresh_list = [ float(x.strip()) for x in args.thresh_list.split(',') ]

    for conf_thresh in thresh_list:

        det_output_dir = osp.join(args.output_dir, '%s_%.2f' % (args.split, conf_thresh))        
        if not osp.exists(det_output_dir):
            os.makedirs(det_output_dir)

        for video_name in video_list:
            print('CONF_THRESH: %.2f, VIDEO: %s' % (conf_thresh, video_name))
            dets_file_name = osp.join(args.det_dir, video_name + '.txt')
            det_dict = face_utils.parse_wider_gt(dets_file_name)
            out_file_name = osp.join(det_output_dir, video_name + '.txt')
            write_formatted_dets(video_name, out_file_name, det_dict, conf_thresh)


        
        
        
