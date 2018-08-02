#!/usr/bin/env python

"""

WT_PATH=Outputs/e2e_faster_rcnn_R-50-C4_1x/Jul30-15-51-27_node097_step/ckpt/model_step79999.pth
CFG_PATH=configs/wider_face/e2e_faster_rcnn_R-50-C4_1x.yaml

srun --pty --mem 50000 --gres gpu:1 -p m40-short \
  python tools/eval/run_face_detection_on_wider.py \
  --cfg ${CFG_PATH} \
  --load_ckpt ${WT_PATH} \
  --exp_name frcnn-R-50-C4-1x



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

# Use a non-interactive backend
import matplotlib
matplotlib.use('Agg')

import numpy as np
import cv2

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
from utils.detectron_weight_helper import load_detectron_weight
from utils.timer import Timer

import matplotlib.pyplot as plt
import scipy.io as sio



def get_imdb_wider(split):
  data_dir = 'data/WIDER'  # fixed (use symlinks)
  list_file = os.path.join(data_dir, 'wider_face_%s_imlist.txt' % split)
  fid = open(list_file, 'r')
  imdb = []
  image_names = []
  for im_name in fid:
    image_names.append(im_name.strip('\n'))
  imdb.append(image_names)
  return imdb


def parse_args():
  """Parse input arguments."""
  parser = argparse.ArgumentParser(description='Face Detection using Faster R-CNN')

  parser.add_argument('--exp_name', required=True, dest='det_dir', help='detector name')
  
  parser.add_argument(
    '--no_cuda', dest='cuda', help='whether use CUDA', action='store_false')
  
  parser.add_argument(
    '--cfg', dest='cfg_file', required=True, help='optional config file')
  parser.add_argument(
    '--set', dest='set_cfgs',
    help='set config keys, will overwrite config in the cfg_file',
    default=[], nargs='+')

  parser.add_argument('--load_ckpt', help='path of checkpoint to load')
  parser.add_argument(
    '--load_detectron', help='path to the detectron weight pickle file')

  parser.add_argument('--split', dest='split', default='val', help='train or val') 
  

  args = parser.parse_args()

  return args


if __name__ == '__main__':
  
  CONF_THRESH = 0.1  # changed from HZJ's 
  NMS_THRESH = 0.15
  cfg.TEST.SCALE = 800
  cfg.TEST.MAX_SIZE = 1333


  if not torch.cuda.is_available():
    sys.exit("Need a CUDA device to run the code.")

  args = parse_args()
  print('Called with args:')
  print(args)

  cfg.MODEL.NUM_CLASSES = 2
  print('load cfg from file: {}'.format(args.cfg_file))
  cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  assert bool(args.load_ckpt) ^ bool(args.load_detectron), \
    'Exactly one of --load_ckpt and --load_detectron should be specified.'
  cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False  # Don't need to load imagenet pretrained weights
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


  data_dir = 'data/WIDER/WIDER_%s/images' % args.split
  det_dir = 'Outputs/evaluations/%s/WIDER-det/WIDER_%s' % (args.det_dir, args.split) # outputs as txt files


  if not os.path.exists(det_dir):
    os.makedirs(det_dir)

  imdb = get_imdb_wider(args.split)  
  image_names = imdb[0]

  for idx, im_name in enumerate(image_names):

    txt_name = os.path.splitext(im_name)[0] + '.txt'
    dir_name, tmp_im_name = os.path.split(im_name)

    im = cv2.imread(os.path.join(data_dir, im_name))
    assert im is not None

    scores, boxes, im_scale, blob_conv = im_detect_bbox(
            net, im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE)

    cls_ind = 1
    cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    dets = np.hstack((cls_boxes,
              cls_scores[:, np.newaxis])).astype(np.float32)

    keep = box_utils.nms(dets, cfg.TEST.NMS)
    dets = dets[keep, :]

    keep = np.where(dets[:, 4] > CONF_THRESH)
    dets = dets[keep]

    dets[:, 2] = dets[:, 2] - dets[:, 0] + 1
    dets[:, 3] = dets[:, 3] - dets[:, 1] + 1

    # Save detection results -- [x y w h score]
    dir_name, tmp_im_name = os.path.split(im_name)
    if not os.path.exists(os.path.join(det_dir, dir_name)):
      os.makedirs(os.path.join(det_dir, dir_name))

    with open(os.path.join(det_dir, txt_name), 'w') as fid:
      fid.write(im_name + '\n') 
      fid.write(str(dets.shape[0]) + '\n')
      for j in xrange(dets.shape[0]):
        fid.write('%f %f %f %f %f\n' % (dets[j, 0], dets[j, 1], 
                                        dets[j, 2], dets[j, 3], 
                                        dets[j, 4]))

    if ((idx + 1) % 100) == 0:
      sys.stdout.write('%.3f%% ' % ((idx + 1) / len(image_names) * 100))
      sys.stdout.flush()


  # os.system('cp ./fddb_res/*.txt /home/hzjiang/Code/FDDB/results')


      