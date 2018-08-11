#!/usr/bin/env python3

"""

Perform inference on frames from a video using Detectron saved checkpoint.
A symlink 'data/CS6' should point to the CS6 data root location
(on Gypsum this is in /mnt/nfs/scratch1/arunirc/data/CS6/CS6/CS6.0.01/CS6).

Usage (on slurm cluster):

srun --pty --mem 100000 --gres gpu:1 python tools/face/detect_video.py \
--vis --video_name 801.mp4

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
# from utils.timer import Timer


DET_NAME = 'frcnn-R-50-C4-1x'
CFG_PATH = 'configs/wider_face/e2e_faster_rcnn_R-50-C4_1x.yaml'
WT_PATH = 'Outputs/e2e_faster_rcnn_R-50-C4_1x/Jul30-15-51-27_node097_step/ckpt/model_step79999.pth'

CONF_THRESH = 0.25
NMS_THRESH = 0.15
OUT_DIR = 'Outputs/evaluations/%s/cs6/sample-baseline-video_conf-%.2f' % (DET_NAME, CONF_THRESH)
# CONF_THRESH = 0.85   
# CONF_THRESH = 0.25   # very low threshold, similar to WIDER eval

VID_NAME = '1100.mp4'
# DATA_DIR = '/mnt/nfs/scratch1/arunirc/data/CS6/CS6/CS6.0.01/CS6'
DATA_DIR = 'data/CS6'


def parse_args():
    parser = argparse.ArgumentParser(description='Detectron inference on video')
    parser.add_argument(
        '--exp_name',
        help='detector name', 
        default=DET_NAME
    )
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='cfg model file (/path/to/model_prototxt)',
        default=CFG_PATH,
        type=str
    )
    parser.add_argument(
        '--load_ckpt',
        help='checkpoints weights model file (/path/to/model_weights.pkl)',
        default=WT_PATH,
        type=str
    )
    parser.add_argument(
    '--load_detectron', help='path to the detectron weight pickle file'
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default=OUT_DIR,
        type=str
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
    # parser.add_argument(
    #     '--save_frame',
    #     dest='save_frame',
    #     help='Save the video frames that have detections',
    #     action='store_true',
    #     default=False
    # )
    parser.add_argument(
        '--data_dir', help='Path to video file', default=DATA_DIR
    )
    parser.add_argument(
        '--video_name', help='Name of video file', default=VID_NAME
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

    # Model setup
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


    # Data setup
    video_path = osp.join(args.data_dir, 'videos', args.video_name)
    if osp.exists(video_path):
        videogen = skvideo.io.vreader(video_path)
    else:
        raise IOError('Path to video not found: \n%s' % video_path)

    vid_name = osp.basename(video_path).split('.')[0]

    if args.vis:
        img_output_dir = osp.join(args.output_dir, vid_name)
        if not osp.exists(img_output_dir):
            os.makedirs(img_output_dir)
        
    
    # Detect faces on video frames
    start = time.time()
    with open(os.path.join(args.output_dir, vid_name + '.txt'), 'w') as fid:
        det_list = []
        for i, im in enumerate(videogen):
            im_name = '%s_%08d' % (vid_name, i)
            print(im_name)

            im = im[:,:,(2,1,0)] # RGB --> BGR

            # Detect faces and regress bounding-boxes
            scores, boxes, im_scale, blob_conv = im_detect_bbox(
            net, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE)

            cls_ind = 1
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                    cls_scores[:, np.newaxis])).astype(np.float32)            
            keep = box_utils.nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            keep = np.where(dets[:, 4] > args.thresh)
            dets = dets[keep]

            dets[:, 2] = dets[:, 2] - dets[:, 0] + 1
            dets[:, 3] = dets[:, 3] - dets[:, 1] + 1


            # Saving visualized frames
            if args.vis:
                viz_out_path = osp.join(img_output_dir, im_name + '.jpg')

            if dets.size == 0: # nothing detected
                if args.vis:
                    cv2.imwrite(viz_out_path, im)
            else:
                if args.vis:
                    im_det = draw_detection_list( im, dets.copy() )
                    cv2.imwrite(viz_out_path, im_det)

                # Writing to text file
                fid.write(im_name + '\n')
                fid.write(str(dets.shape[0]) + '\n')
                for j in xrange(dets.shape[0]):
                    fid.write('%f %f %f %f %f\n' % ( dets[j, 0], dets[j, 1], 
                                                     dets[j, 2], dets[j, 3], 
                                                     dets[j, 4]) )

            
            # if ((i + 1) % 100) == 0:
            #     sys.stdout.write('%d ' % i)
            #     sys.stdout.flush()

    end = time.time()
    print('Execution time in seconds: %f' % (end - start))
        
        
        
