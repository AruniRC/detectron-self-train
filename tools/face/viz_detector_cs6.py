#!/usr/bin/env python3

"""

Visualize a detector output on the CS6 validation set. 
The val set GT annotations are in an FDDB/WIDER-style txt file format.

A symlink 'data/CS6' should point to the CS6 data root location
(on Gypsum this is in /mnt/nfs/scratch1/arunirc/data/CS6/CS6/CS6.0.01/CS6).

Usage (on slurm cluster):

srun --mem 10000 python tools/face/viz_detector_cs6.py

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
import skvideo
import skvideo.io

import sys
sys.path.append('./tools')

import torch
import torch.nn as nn
from torch.autograd import Variable

import _init_paths
import nn as mynn
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from core.test import im_detect_bbox
from modeling.model_builder import Generalized_RCNN
import datasets.dummy_datasets as datasets
import utils.boxes as box_utils  # for NMS
import utils.misc as misc_utils
import utils.net as net_utils
import utils.vis as vis_utils
import utils.face_utils as face_utils
from utils.detectron_weight_helper import load_detectron_weight


# set random seeds
np.random.seed(999)
random.seed(999)
torch.cuda.manual_seed_all(999)
torch.manual_seed(999)
torch.backends.cudnn.deterministic = True


# --- Quick settings ---
GT_FILE = 'data/CS6_annot/annot-format-GT/cs6_gt_annot_val-easy.txt'
OUT_DIR = 'Outputs/visualizations/'
NUM_IMG = 100

DET_NAME = 'baseline-cs6'
CFG_PATH = 'configs/wider_face/e2e_faster_rcnn_R-50-C4_1x.yaml'
WT_PATH = 'Outputs/e2e_faster_rcnn_R-50-C4_1x/Jul30-15-51-27_node097_step/ckpt/model_step79999.pth'
CONF_THRESH = 0.25
NMS_THRESH = 0.15
DATA_DIR = 'data/CS6_annot' 

def parse_args():
    parser = argparse.ArgumentParser(description='Creating CS6 ground truth data')
    parser.add_argument(
        '--det_name',
        help='detector name', 
        default=DET_NAME
    )
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='cfg model file (/path/to/model_prototxt)',
        default=CFG_PATH,
    )
    parser.add_argument(
        '--load_ckpt',
        help='checkpoints weights model file (/path/to/model_weights.pkl)',
        default=WT_PATH,
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='Threshold on class score (default: 0.5)',
        default=CONF_THRESH,
        type=float
    )
    parser.add_argument(
        '--output_dir', help='directory for saving outputs',
        default=OUT_DIR,
    )
    parser.add_argument(
        '--gt_file', help='Name of dataset file in FDDB-format', 
        default=GT_FILE
    )
    parser.add_argument(
        '--imdir', help="root directory for loading dataset images",
        default=DATA_DIR
    )
    parser.add_argument(
        '--data-dir',
        dest='data_dir', 
        help='Path to CS6 annotations folder', 
        default=DATA_DIR
    )
    parser.add_argument(
        '--num_im', help='Number of images to visualize per video', 
        default=NUM_IMG, type=int
    )
    return parser.parse_args()



_GREEN = (18, 127, 15)
# ------------------------------------------------------------------------------
def draw_detection_list(im, dets):
# ------------------------------------------------------------------------------
    """ Draw detected bounding boxes on a copy of image and return it.
        [x0 y0 w h conf_score]
    """
    im_det = im.copy()
    if dets.ndim == 1:
        dets = dets[np.newaxis,:] # handle single detection case

    # format into [xmin, ymin, xmax, ymax]
    dets[:, 2] = dets[:, 2] + dets[:, 0]
    dets[:, 3] = dets[:, 3] + dets[:, 1]

    for i, det in enumerate(dets):
        bbox = dets[i, :4]
        conf_score = dets[i, 4]
        x0, y0, x1, y1 = [int(x) for x in bbox]
        line_color = _GREEN
        cv2.rectangle(im_det, (x0, y0), (x1, y1), line_color, thickness=2)
        disp_str = '%d: %.2f' % (i, conf_score)
        face_utils._draw_string(im_det, (x0, y0), disp_str)

    return im_det




if __name__ == '__main__':
    

    args = parse_args()

    det_dict = face_utils.parse_wider_gt(args.det_file)

    out_dir = osp.join(args.output_dir, 
                       osp.splitext(osp.basename(args.gt_file))[0], 
                       args.det_name)
    if not osp.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    i = 0
    for (image_name, dets) in det_dict.items():
        if len(dets) == 0:
            continue
        print(image_name)
        im = cv2.imread(osp.join(args.imdir, image_name))
        assert im.size > 0
        im_det = draw_detection_list(im, np.array(dets))
        out_path = osp.join(out_dir, image_name.replace('/', '_'))
        cv2.imwrite(out_path, im_det)
        i += 1 
        if i == args.num_im:
            break

    print('Done visualizing')
    print('Results folder: %s' % out_dir)



    

        
        
        
        





