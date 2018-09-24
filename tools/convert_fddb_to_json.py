import os
import sys
import cv2
import numpy as np
import json

import _init_paths
from utils.face_utils import parse_wider_gt
import core

dataset_root = '/mnt/nfs/work1/elm/arunirc/Research/detectron-video/mask-rcnn.pytorch/data/CS6_annot'

gt_fddb = '/mnt/nfs/work1/elm/arunirc/Research/detectron-video/detectron_distill/Detectron-pytorch-video/data/CS6_annot/annot-format-GT/cs6_gt_annot_val-easy.txt'

det_fddb = '/mnt/nfs/work1/elm/arunirc/Research/detectron-video/mask-rcnn.pytorch/Outputs/evaluations/frcnn-R-50-C4-1x/cs6/train-WIDER_val-easy_conf-0.1/annot-format-dets/cs6_annot_eval_scores.txt'

fddb_file = det_fddb
json_file = 'det_cs6.json'

def convert_to_json(fddb_file):
    ds_json = {'images' : [],
               'categories' : [],
               'annotations' : []
              }
    ann = parse_wider_gt(fddb_file)
    bbox_id = 0
    for fid,filename in enumerate(ann.keys()):
        #img = cv2.imread(os.path.join(dataset_root,filename))
        #w,h,c = img.shape
        w,h,c = (720,1280,3)
        im_prop = {'width':w,
                   'height':h,
                   'id':fid,
                   'file_name':filename
                  }
        ds_json['images'].append(im_prop)
        bboxes = ann[filename]
        print('Reading:',filename)
        for bbox in bboxes:
            bbox_prop = {'id':bbox_id,
                         'image_id':fid,
                         'segmentation':[],
                         'category_id':1,
                         'iscrowd':0,
                         'bbox':bbox,
                         'area':bbox[2]*bbox[3]
                        }
            ds_json['annotations'].append(bbox_prop)
            bbox_id += 1
    print('Total:',fid,'files saved in JSON format')
    with open(json_file,'w') as f:
        json.dump(ds_json,f)
    f.close()

if __name__ == '__main__':
    convert_to_json(fddb_file)


