#!/usr/bin/env python

"""

Perform inference on pre-extracted frame images in CS6 using Detectron.
A symlink 'data/CS6' should point to the CS6 data root location
(on Gypsum this is in /mnt/nfs/scratch1/arunirc/data/CS6/CS6/CS6.0.01/CS6). 
A sylmlink 'data/CS6_annot' should point to the formatted anntotations and 
extracted frames from CS6, created by running "tools/make_ground_truth_data.py" 
The test/val splits and image lists are pre-defined.
TODO - make the splits lists available online at a fixed location (arc). 

This code is primarily used for quicker visualizations, when we do not need to  
run the detector on *every* frame of a video, but can just read in those  
frames (pre-extracted and saved as images) used by the Validation or Test set.

Usage (on slurm cluster):

srun --pty --mem 100000 --gres gpu:1 python tools/face/viz_detect_images.py

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
import json

# Use a non-interactive backend
import matplotlib
matplotlib.use('Agg')

import numpy as np
import cv2
import skvideo
import skvideo.io

import torch
import torch.nn as nn
from torch.autograd import Variable

import sys
sys.path.append('./tools')

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

# fixing seeds
np.random.seed(999)
# random.seed(999)
torch.cuda.manual_seed_all(999)
torch.manual_seed(999)
torch.backends.cudnn.deterministic = True


DEBUG = False

"""
# # --- Quick settings ---
# DET_NAME = 'frcnn-R-50-C4-1x_Baseline'
# CFG_PATH = 'configs/wider_face/e2e_faster_rcnn_R-50-C4_1x.yaml'
# WT_PATH = 'Outputs/e2e_faster_rcnn_R-50-C4_1x/Jul30-15-51-27_node097_step/ckpt/model_step79999.pth'

# --- CS6-train-DET + WIDER ---
# DET_NAME = 'frcnn-R-50-C4-1x_CS6-train-Det+WIDER'
# CFG_PATH = 'configs/cs6/e2e_faster_rcnn_R-50-C4_1x_4gpu_joint_bs64_30k.yaml'
# WT_PATH = 'Outputs/e2e_faster_rcnn_R-50-C4_1x_4gpu_joint_bs64_30k/Oct10-20-16-39_node149_step/ckpt/model_step9999.pth'

# # --- CS6-train-HP + WIDER ---
# DET_NAME = 'frcnn-R-50-C4-1x_CS6-train-HP+WIDER'
# CFG_PATH = 'configs/cs6/e2e_faster_rcnn_R-50-C4_1x_4gpu_joint_bs64_30k.yaml'
# WT_PATH = 'Outputs/e2e_faster_rcnn_R-50-C4_1x_4gpu_joint_bs64_30k/Sep25-13-48-15_node137_step/ckpt/model_step4999.pth'

# --- CS6-train-HP-distill-0.7 + WIDER ---
# DET_NAME = 'frcnn-R-50-C4-1x_CS6-train-HP-distill-0.7+WIDER'
# CFG_PATH = 'configs/cs6/e2e_faster_rcnn_R-50-C4_1x_4gpu_distill-07.yaml'
# WT_PATH = 'Outputs/e2e_faster_rcnn_R-50-C4_1x_4gpu_distill-07/Oct03-22-00-34_node142_step/ckpt/model_step4999.pth'

DATASET = 'cs6-val-easy'
GT_FILE = 'data/CS6_annot/annot-format-GT/cs6_gt_annot_val-easy.txt'
OUT_DIR = 'Outputs/visualizations/'
NUM_IMG = 100 
CONF_THRESH = 0.25
NMS_THRESH = 0.15
DATA_DIR = 'data/CS6_annot'   # this points to ANNOTS folder, and *not* the raw CS6 protocol
"""
 
# -- BDD Dets18k only --- #
DET_NAME = 'bdd_dets18k_only'
CFG_PATH = 'configs/baselines/cityscapes.yaml'
WT_PATH = 'Outputs/cityscapes/bdd_dets18k_only/ckpt/model_step64999.pth'

# -- BDD HP18k only -- #
#DET_NAME = 'bdd_hp18k_only'
#CFG_PATH = 'configs/baselines/cityscapes.yaml'
#WT_PATH = 'Outputs/cityscapes/bdd_hp18k_only/ckpt/model_step64999.pth'

# -- BDD source+Dets18k -- #
#DET_NAME = 'bdd_source_and_dets18k'
#CFG_PATH = 'configs/baselines/cityscapes.yaml'
#WT_PATH = 'Outputs/cityscapes/bdd_source_and_dets18k/ckpt/model_step64999.pth'

# -- BDD source+HP18k -- #
#DET_NAME = 'bdd_source_and_hp18k'
#CFG_PATH = 'configs/baselines/cityscapes.yaml'
#WT_PATH = 'Outputs/cityscapes/bdd_source_and_hp18k/ckpt/model_step64999.pth'

# -- BDD source+HP18k distill 0.3 -- #


DATASET = 'bdd_peds_not_clear_any_daytime_val'
GT_FILE = 'data/CS6_annot/annot-format-GT/cs6_gt_annot_val-easy.txt'
OUT_DIR = 'Outputs/visualizations/'
NUM_IMG = 100
CONF_THRESH = 0.25
NMS_THRESH = 0.15
DATA_DIR = 'data/bdd100k'   # this points to ANNOTS folder, and *not* the raw CS6 protocol


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--annot_file', help='Name of dataset file in FDDB-format', 
        default=GT_FILE
    )
    parser.add_argument(
        '--dataset', help='Name of dataset and split', 
        default=DATASET
    )
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
    '--load_detectron', help='path to the detectron weight pickle file'
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default=OUT_DIR,
    )
    parser.add_argument(
        '--no_cuda', dest='cuda', help='whether use CUDA', 
        action='store_false'
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='Threshold on class score (default: 0.5)',
        default=CONF_THRESH,
        type=float
    )
    parser.add_argument(
        '--vis',
        dest='vis',
        help='Visualize detections on video frames option',
        action='store_true',
        default=False
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
# -----------------------------------------------------------------------------------
def draw_detection_list(im, dets):
# -----------------------------------------------------------------------------------
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

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    args = parse_args()
    # args.output_dir = args.output_dir % args.exp_name
    print('Called with args:')
    print(args)
    
    # -----------------------------------------------------------------------------------
    # Model setup
    # -----------------------------------------------------------------------------------
    cfg.TEST.SCALE = 800
    cfg.TEST.MAX_SIZE = 1333
    cfg.MODEL.NUM_CLASSES = 2
    cfg.TEST.NMS = NMS_THRESH
    print('load cfg from file: {}'.format(args.cfg_file))
    cfg_from_file(args.cfg_file)

    assert bool(args.load_ckpt) ^ bool(args.load_detectron), \
        'Exactly one of --load_ckpt and --load_detectron should be specified.'
    cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False
    assert_and_infer_cfg()

    net = Generalized_RCNN()

    if args.cuda:
        net.cuda()

    if args.load_ckpt:
        load_name = args.load_ckpt
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(net, checkpoint['model'])

    if args.load_detectron:
        print("loading detectron weights %s" % args.load_detectron)
        load_detectron_weight(net, args.load_detectron)

    net = mynn.DataParallel(net, cpu_keywords=['im_info', 'roidb'],
                            minibatch=True, device_ids=[0])  # only support single GPU
    net.eval()


    # -----------------------------------------------------------------------------------
    # Data setup
    # -----------------------------------------------------------------------------------
    annot_file = args.annot_file
    annot_dict = face_utils.parse_wider_gt(annot_file)
    image_subset = np.array(list(annot_dict.keys()))
    np.random.shuffle(image_subset)
    image_subset = image_subset[:args.num_im]

    # output folders
    img_output_dir = osp.join(args.output_dir, args.det_name + '_' + args.dataset)
    if not osp.exists(img_output_dir):
        os.makedirs(img_output_dir)

    with open(osp.join(args.output_dir, 'config_args.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=4, sort_keys=True)
    
    
    # -----------------------------------------------------------------------------------
    # Detect faces on CS6 frames
    # -----------------------------------------------------------------------------------
    start = time.time()
    for i, im_name in enumerate(image_subset):

        print('%d/%d: %s' % (i, len(image_subset), im_name))
        im = cv2.imread(osp.join(args.data_dir, im_name))
        assert im.size != 0

        # Detect faces and regress bounding-boxes
        scores, boxes, im_scale, blob_conv = im_detect_bbox(
            net, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE)

        # Format the detection output
        cls_ind = 1
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                cls_scores[:, np.newaxis])).astype(np.float32)
        keep = box_utils.nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        keep = np.where(dets[:, 4] > CONF_THRESH)
        dets = dets[keep]
        # (x, y, w, h)
        dets[:, 2] = dets[:, 2] - dets[:, 0] + 1
        dets[:, 3] = dets[:, 3] - dets[:, 1] + 1
        print('Num. detections: %d' % dets.shape[0])
        # if dets.size == 0: # nothing detected
        #     continue

        # Saving visualized frames
        viz_out_path = osp.join( img_output_dir, osp.basename(im_name) ) 
        if dets.size != 0: 
            im_det = draw_detection_list( im, dets.copy() )
            cv2.imwrite(viz_out_path, im_det)
        else:
            cv2.imwrite(viz_out_path, im) # nothing detected, save orig img
        
        if ((i + 1) % 100) == 0:
            sys.stdout.write('%d ' % i)
            sys.stdout.flush()

    end = time.time()
    print('Execution time in seconds: %f' % (end - start))
    print('Finished: SUCCESS')
        
        
        

