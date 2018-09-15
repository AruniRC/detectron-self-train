#!/usr/bin/env python

"""
Convert the "CS6 format" video GT to a single annotations format file. 
A symlink 'data/CS6' should point to the CS6 data root location. This script is 
called for each split of CS6 to create a single annotations file from the 
multiple gt annot files under 'data/CS6_Annot/video_annots/'. 

By default the output is saved as:
    'data/CS6_Annot/annot-format-GT/cs6_gt_annot_<split-name>.txt'

<split-name> should be one of: 
    'train-subset'  --  merge annots from 19 videos in "Train-subset"
    'train'         --  merge annotations from 88 videos in "Train"



TODO: Usage (on slurm cluster):

srun --pty --mem 50000 python tools/face/convert_....

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
from pprint import pprint

import numpy as np
import cv2
from PIL import Image

import sys
sys.path.append('./tools')

import _init_paths
import utils.face_utils as face_utils

# ------------------------------------------------------------------------------
#   Quick settings
# ------------------------------------------------------------------------------
DET_DIR = 'data/CS6_annot/video_annots'


# CS6 Train-subset GT
# SPLIT = 'train-subset'
# IS_SUBSET = True
# VIDEO_LIST_FILE = 'list_video_train_subset.txt'  # parent folder is 'data/CS6'

# CS6 Train GT
SPLIT = 'test-easy'
IS_SUBSET = False
VIDEO_LIST_FILE = 'list_video_test_easy.txt'  # parent folder is 'data/CS6'

# OUT_DIR = 'Outputs/evaluations/%s/cs6/mining-detections'  # usually unchanged
# ------------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert the "CS6 format" detections to the "WIDER annots" format.'
    )
    parser.add_argument(
        '--output_dir',
        help='Saves detection txt files in WIDER annot format',
        default='',
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
    parser.add_argument(
        '--subset', 
        help='Flag for subset', 
        action='store_true',
        default=IS_SUBSET)

    return parser.parse_args()


def load_all_gt(video_list, det_dir):
    ''' Helper function. Load all detections (txt) and merge into single dict. '''
    det_dict_all = {}
    for video_name in video_list:
        dets_file_name = osp.join(det_dir, video_name + '.txt')
        det_dict = face_utils.parse_wider_gt(dets_file_name)
        det_dict_all.update(det_dict)
    return det_dict_all


def write_annot_gt(out_file_name, det_dict):
    '''
    Output dets format: 
        <image-path>
        <num-dets>
        [x1, y1, w, h]
        [x1, y1, w, h]
        ....

    '''
    with open(out_file_name, 'w') as fid:
        for im_name, dets in det_dict.items():
            dets = np.array(dets)
            if dets.shape[0] == 0:
                continue
            fid.write(im_name + '\n')
            fid.write(str(dets.shape[0]) + '\n')
            for j in xrange(dets.shape[0]):
                fid.write('%f %f %f %f\n' % ( dets[j, 0], dets[j, 1], 
                                              dets[j, 2], dets[j, 3]) )


if __name__ == '__main__':

    args = parse_args()

    # Create output folder
    if not args.output_dir:
        args.output_dir = osp.abspath('data/CS6_annot/annot-format-GT' )
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print('Called with args:')
    pprint(args)

    # List of video files in Training set/subset (strips .ext)
    with open( osp.join('data/CS6', args.video_list_file), 'r' ) as f:
        video_list = [ y.split('.')[0] for y in [ x.strip() for x in f ] ]

    # Load all CS6 ground-truth for that video list into a dict
    det_dict = load_all_gt(video_list, args.det_dir)
    assert(len(det_dict.keys()) != 0)

    # Write all GT as "hard-annot" into a file
    out_file_name = osp.join(args.output_dir, 'cs6_gt_annot_%s.txt' % args.split)

    print('Writing detections to "annot" file: %s' % out_file_name)
    write_annot_gt(out_file_name, det_dict)
    print('Done.')
