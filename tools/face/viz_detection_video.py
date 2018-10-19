#!/usr/bin/env python3

"""

Visualize detections from an FDDBstyle detection file for a video.

A symlink 'data/CS6' should point to the CS6 data root location
(on Gypsum this is in /mnt/nfs/scratch1/arunirc/data/CS6/CS6/CS6.0.01/CS6).

Usage (on slurm cluster):

srun --mem 10000 python tools/face/viz_detection.py

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
import _init_paths
import utils.face_utils as face_utils


# --- Quick settings ---
MODEL = 'train-CS6-Det-all-5k_val-easy_conf-0.1'
DET_FILE = 'Outputs/evaluations/frcnn-R-50-C4-1x/cs6/' + MODEL + '/eval-dets_val_easy/614.txt'
OUT_DIR = 'Outputs/visualizations/' + MODEL
NUM_IMG = 50



def parse_args():
    parser = argparse.ArgumentParser(description='Creating CS6 ground truth data')
    parser.add_argument(
        '--output_dir', help='directory for saving outputs',
        default=OUT_DIR,
    )
    parser.add_argument(
        '--det_file', help='Name of detections file in FDDB-format', 
        default=DET_FILE
    )
    parser.add_argument(
        '--imdir', help="root directory for loading dataset images",
        default='data/CS6_annot'
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
                       osp.splitext(osp.basename(args.det_file))[0])
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



    

        
        
        
        





