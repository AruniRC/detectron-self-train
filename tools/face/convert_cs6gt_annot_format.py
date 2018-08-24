#!/usr/bin/env python

"""
Convert the "CS6 format" video GT to the WIDER annotations format. 
A symlink 'data/CS6' should point to the CS6 data root location. 

CS6 VIDEO GT FORMAT:
<video-name>_<0-starting-frame-number>
<num-dets>
[x1, y1, w, h, score]
[x1, y1, w, h, score]
... 

WIDER ANNOT FORMAT (hard threshold on the det-scores):
<image-path>
<num-dets>
[x1, y1, w, h]
[x1, y1, w, h]
... 

SOFT LABELS FORMAT:
<image-path>
<num-dets>
[x1, y1, w, h, score]
[x1, y1, w, h. score]
... 

The format of <image-path>: 
    frames/<vid-name>/<vid-name>_<frame-num>.jpg

Output:
=================

DET_FOLDER (given)

FILES:
    DET_FOLDER/annot-format-dets/
        cs6_annot_train_subset_scores.txt
        cs6_annot_train_subset_conf-0.5.txt
        .... 


TODO: Usage (on slurm cluster):

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
from pprint import pprint

import numpy as np
import cv2
from PIL import Image

import sys
sys.path.append('./tools')

import _init_paths
import utils.face_utils as face_utils



DET_NAME = 'gt'
DET_DIR = 'data/CS6_annot/video_annots'
VIDEO_LIST_FILE = 'list_video_train_subset.txt'  # parent folder is 'data/CS6'
CONF_THRESH_LIST = '0.5'  # try one threshold for now
SPLIT = 'train-subset'
IS_SUBSET = True
# OUT_DIR = 'Outputs/evaluations/%s/cs6/mining-detections'  # usually unchanged


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert the "CS6 format" detections to the "WIDER annots" format.'
    )
    parser.add_argument(
        '--exp_name',
        help='detector name', 
        default=DET_NAME
    )
    parser.add_argument(
        '--output_dir',
        help='Saves detection txt files in WIDER annot format',
        default='',
    )
    parser.add_argument(
        '--thresh_list',
        help='Comma-separated list of thresholds on class score',
        default=CONF_THRESH_LIST,
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


def get_extracted_imlist(imlist_file):
    ''' Return set of (pre-existing) images listed in imlist file. '''
    with open(imlist_file, 'r' ) as f:
        im_list = [ y[0] for y in \
                    list(map(osp.splitext, 
                    list(map(osp.basename, 
                    [ x.strip() for x in f ]) 
                    )))
                  ]
    im_frame_set = set(im_list)
    return im_frame_set


def prune_extra_images(det_dict, det_frame_set, im_frame_set):
    # Prune images not in `im_frame_set`
    extra_dets = list(det_frame_set - im_frame_set)
    for frame_name in extra_dets:
        det_dict.pop(frame_name, None)
    assert len(det_frame_set) - len(extra_dets) \
            == len(set(det_dict.keys())) # sanity-check



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
            # vid_name = im_name.split('_')[0] # im_name format: <vid-name>_<frame-num>
            # im_path = 'frames/%s/%s.jpg' % (vid_name, im_name)
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


    ## Not needed for GT
    # Load list of CS6 frame images
    #   -- we keep only training images that have existing frame images
    #   -- (this avoids extracting more images at marginal cost in terms of data)
    # im_list = osp.join('data/CS6_annot', 'cs6_im_list.txt')
    # im_frame_set = get_extracted_imlist(im_list)


    # Load all CS6 ground-truth for that video list into a dict
    det_dict = load_all_gt(video_list, args.det_dir)
    # det_frame_set = set(det_dict.keys())
    assert(len(det_dict.keys()) != 0)

    # Write all GT as "hard-annot" into a file
    out_file_name = osp.join(args.output_dir, 'cs6_gt_annot_%s.txt' % args.split)

    print('Writing detections to "annot" file: %s' % out_file_name)
    write_annot_gt(out_file_name, det_dict)
    print('Done.')
