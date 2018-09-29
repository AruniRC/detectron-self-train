#!/usr/bin/env python

"""
Convert the "CS6 format" video detections to the WIDER annotations format. 
A symlink 'data/CS6' should point to the CS6 data root location. 
(This generated dets file can then be converted into the trianing JSON using 
lib/datasets/wider/convert_coco....)

Two types of lists are generated - one containing the "soft-labels" (SCORES) and  
the other containing just the detections as positive ground-truth.
By default the output files are saved in a sub-folder created under DET_DIR.

CS6 VIDEO DET FORMAT:
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


Usage (on slurm cluster):

srun --pty --mem 50000 --gres gpu:1 python tools/face/convert_dets_annot_format.py ...

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


# Quick-specify settings:


# ------------------------------------------------------------------------------
#   For CS6 Train - Detections as pseudo-labels
# ------------------------------------------------------------------------------
# DET_NAME = 'frcnn-R-50-C4-1x'
# DET_DIR = 'Outputs/evaluations/frcnn-R-50-C4-1x/cs6/train-WIDER_train-video_conf-0.25/'
# VIDEO_LIST_FILE = 'list_video_train.txt'  # parent folder is 'data/CS6'
# CONF_THRESH_LIST = '0.5'    # comma-separated string of thresholds
# SPLIT = 'train'
# IS_SUBSET = False


# ------------------------------------------------------------------------------
#   For CS6 Train Easy - Hard Positives
# ------------------------------------------------------------------------------
# IS_HARD_EX = True
# DET_NAME = 'frcnn-R-50-C4-1x'
# DET_DIR = 'Outputs/tracklets/hp-res-cs6/'
# VIDEO_LIST_FILE = 'list_video_train_easy.txt'
# CONF_THRESH_LIST = '0.5'
# IS_SUBSET = False
# SPLIT = 'train_easy'


# ------------------------------------------------------------------------------
#   For CS6 Val Easy
# ------------------------------------------------------------------------------
DET_NAME = 'frcnn-R-50-C4-1x'
DET_DIR = 'Outputs/evaluations/frcnn-R-50-C4-1x/cs6/train-CS6-GT-all-30k_val-easy_conf-0.1/'
VIDEO_LIST_FILE = 'list_video_val_easy.txt'  # parent folder is 'data/CS6'
CONF_THRESH_LIST = '0.1'    # comma-separated string of thresholds
SPLIT = 'val_easy'
IS_SUBSET = False
IS_HARD_EX = False



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
    parser.add_argument(
        '--hard_ex', 
        help='Flag for hard examples', 
        action='store_true',
        default=IS_HARD_EX)

    return parser.parse_args()


def load_all_dets(video_list, det_dir):
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



def write_hard_annot_dets(out_file_name, det_dict, conf_thresh):
    '''
    Write the detections, scores *thresholded*, to a text file.

    Output dets format: 
        <image-path>
        <num-dets>
        [x1, y1, w, h]
        [x1, y1, w, h]
        ....

    The format of <image-path>: 
        frames/<vid-name>/<vid-name>_<frame-num>.jpg

    '''
    with open(out_file_name, 'w') as fid:
        for im_name, dets in det_dict.items():
            dets = np.array(dets)
            if dets.shape[0] == 0:
                continue
            keep = np.where(dets[:, 4] > conf_thresh)
            dets = dets[keep]
            if dets.shape[0] == 0:
                continue

            vid_name = im_name.split('_')[0] # im_name format: <vid-name>_<frame-num>
            im_path = 'frames/%s/%s.jpg' % (vid_name, im_name)
            fid.write(im_path + '\n')
            fid.write(str(dets.shape[0]) + '\n')
            for j in xrange(dets.shape[0]):
                fid.write('%f %f %f %f\n' % ( dets[j, 0], dets[j, 1], 
                                              dets[j, 2], dets[j, 3]) )



def write_scores_dets(out_file_name, det_dict):
    '''
    Write the detections, scores included, to a text file.

    Output dets format: 
        <image-path>
        <num-dets>
        [x1, y1, w, h, score]
        [x1, y1, w, h, score]
        ....

    The format of <image-path>: 
        frames/<vid-name>/<vid-name>_<frame-num>.jpg

    '''
    with open(out_file_name, 'w') as fid:
        for im_name, dets in det_dict.items():
            dets = np.array(dets)
            if dets.shape[0] == 0:
                continue  # skip images with no detections

            vid_name = im_name.split('_')[0] # im_name format: <vid-name>_<frame-num>
            im_path = 'frames/%s/%s.jpg' % (vid_name, im_name)
            fid.write(im_path + '\n')
            fid.write(str(dets.shape[0]) + '\n')
            for j in xrange(dets.shape[0]):
                fid.write('%f %f %f %f %f\n' % ( dets[j, 0], dets[j, 1], 
                                                 dets[j, 2], dets[j, 3], 
                                                 dets[j, 4]) )



def write_hard_ex_dets(out_file_name, det_dict):
    '''
    Write the detections, scores included, to a text file.

    Output dets format: 
        <image-path>
        <num-dets>
        [x1, y1, w, h, score, source]
        [x1, y1, w, h, score, source]
        ....

    The format of <image-path>: 
        frames/<vid-name>/<vid-name>_<frame-num>.jpg

    '''
    with open(out_file_name, 'w') as fid:
        for im_name, dets in det_dict.items():
            dets = np.array(dets)
            if dets.shape[0] == 0:
                continue  # skip images with no detections

            vid_name = im_name.split('_')[0] # im_name format: <vid-name>_<frame-num>
            im_path = 'frames/%s/%s.jpg' % (vid_name, im_name)
            fid.write(im_path + '\n')
            fid.write(str(dets.shape[0]) + '\n')
            for j in xrange(dets.shape[0]):
                fid.write('%f %f %f %f %f\n' % ( dets[j, 0], dets[j, 1], 
                                                 dets[j, 2], dets[j, 3], 
                                                 dets[j, 4]) )




if __name__ == '__main__':

    args = parse_args()

    # Create output folder
    if not args.output_dir:
        args.output_dir = osp.abspath(
                            osp.join(args.det_dir, 'annot-format-dets' ))
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print('Called with args:')
    pprint(args)

    # List of video files in Training set/subset (strips .ext)
    with open( osp.join('data/CS6', args.video_list_file), 'r' ) as f:
        video_list = [ y.split('.')[0] for y in [ x.strip() for x in f ] ]    

    # List of detector conf. thresholds
    thresh_list = [ float(x.strip()) for x in args.thresh_list.split(',') ]

    # Load list of CS6 frame images
    #   -- we keep only training images that have existing frame images
    #   -- (this avoids extracting more images at marginal cost in terms of data)
    im_list = osp.join('data/CS6_annot', 'cs6_im_list.txt')
    im_frame_set = get_extracted_imlist(im_list)

    # Load all detections into a dict
    det_dict = load_all_dets(video_list, args.det_dir)
    det_frame_set = set(det_dict.keys())

    prune_extra_images(det_dict, det_frame_set, im_frame_set)

    # --------------------------------------------------------------------------
    # SOFT-LABELS (SCORES) DETECTIONS
    # Write all detections into an annot file (scores as soft-labels)
    # --------------------------------------------------------------------------
    if args.subset:
        out_file_name = osp.join(args.output_dir, 
                                 'cs6_annot_train_subset_scores.txt')
    else:
        if 'val' in args.split:
            out_file_name = osp.join(args.output_dir, 
                                 'cs6_annot_eval_scores.txt')
        else:
            out_file_name = osp.join(args.output_dir, 
                                 'cs6_annot_train_scores.txt')

    print('Writing detections to "score-annot" file: %s' % out_file_name)
    write_scores_dets(out_file_name, det_dict)
    print('Done.')
    # TODO - print a little summary of num. dets. num. images.


    # --------------------------------------------------------------------------
    # HARD-LABELS DETECTIONS
    # Write detections as "hard-annot" into a file (*thresholded* scores)
    # --------------------------------------------------------------------------
    for conf_thresh in thresh_list:
        if args.subset:
            thresh_file_name = osp.join(args.output_dir, 
                            'cs6_annot_train_subset_conf-%.2f.txt' % conf_thresh)
        else:
            thresh_file_name = osp.join(args.output_dir, 
                            'cs6_annot_train_conf-%.2f.txt' % conf_thresh)

        print('Writing detections to "score-annot" file: %s' % thresh_file_name)
        write_hard_annot_dets(thresh_file_name, det_dict, conf_thresh)
        print('Done.')
