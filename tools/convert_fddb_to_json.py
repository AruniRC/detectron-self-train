import os
import sys
import cv2
import json
import tempfile
import numpy as np

import _init_paths
from utils.face_utils import parse_wider_gt
import core

dataset_root = '/mnt/nfs/work1/elm/arunirc/Research/detectron-video/mask-rcnn.pytorch/data/CS6_annot'

# CS6 ground truth
gt_fddb = '/mnt/nfs/work1/elm/arunirc/Research/detectron-video/detectron_distill/Detectron-pytorch-video/data/CS6_annot/annot-format-GT/cs6_gt_annot_val-easy.txt'
gt_json = '/mnt/nfs/work1/elm/pchakrabarty/cs6_jsons/cs6_gt_annot_val-easy.json'

# WIDER baseline model detections
wider_model_det_fddb = '/mnt/nfs/work1/elm/arunirc/Research/detectron-video/mask-rcnn.pytorch/Outputs/evaluations/frcnn-R-50-C4-1x/cs6/train-WIDER_val-easy_conf-0.1/eval-dets_val_easy'
wider_model_det_json = '/mnt/nfs/work1/elm/pchakrabarty/cs6_jsons/train-WIDER_val-easy_conf-0.1_cs6_annot_eval_scores.json'

# CS6 GT model detections
cs6_gt_model_det_fddb = '/mnt/nfs/work1/elm/arunirc/Research/detectron-video/mask-rcnn.pytorch/Outputs/evaluations/frcnn-R-50-C4-1x/cs6/train-CS6-GT-all-30k_val-easy_conf-0.1/eval-dets_val_easy'
cs6_gt_model_det_json = '/mnt/nfs/work1/elm/pchakrabarty/cs6_jsons/train-CS6-GT-all-30k_val-easy_conf-0.1_cs6_annot_eval_scores.json'

# CS6 HP+WIDER model (bs64-5k) detections
cs6_hp_and_wider_bs64_model_det_fddb = '/mnt/nfs/work1/elm/arunirc/Research/detectron-video/mask-rcnn.pytorch/Outputs/evaluations/frcnn-R-50-C4-1x/cs6/train-CS6-train-HP+WIDER-bs64-5k_val-easy_conf-0.1/eval-dets_val_easy'
cs6_hp_and_wider_bs64_model_det_json = '/mnt/nfs/work1/elm/pchakrabarty/cs6_jsons/train-CS6-HP+WIDER-bs64-5k_val-easy_conf-0.1_cs6_annot_eval_scores.json'

# CS6 train HP+WIDER (bs512-15k) model detections
cs6_hp_and_wider_bs512_model_det_fddb = '/mnt/nfs/work1/elm/arunirc/Research/detectron-video/mask-rcnn.pytorch/Outputs/evaluations/frcnn-R-50-C4-1x/cs6/train-CS6-train-HP+WIDER-bs512-15k_val-easy_conf-0.1/eval-dets_val_easy'
cs6_hp_and_wider_bs512_model_det_json = '/mnt/nfs/work1/elm/pchakrabarty/cs6_jsons/train-CS6-HP+WIDER-bs512-15k_val-easy_conf-0.1_cs6_annot_eval_scores.json'

# CS6 train HP+WIDER (bs512-gpu4-5k) model detections
cs6_hp_and_wider_bs512_gpu4_5k_model_det_fddb = '/mnt/nfs/work1/elm/arunirc/Research/detectron-video/mask-rcnn.pytorch/Outputs/evaluations/frcnn-R-50-C4-1x/cs6/train-CS6-train-HP+WIDER-bs512-gpu4-5k_val-easy_conf-0.1/eval-dets_val_easy'
cs6_hp_and_wider_bs512_gpu4_5k_model_det_json = '/mnt/nfs/work1/elm/pchakrabarty/cs6_jsons/train-CS6-HP+WIDER-bs512-gpu4-5k_val-easy_conf-0.1_cs6_annot_eval_scores.json'

# CS6 train HP
cs6_hp_5k_model_det_fddb = '/mnt/nfs/work1/elm/arunirc/Research/detectron-video/mask-rcnn.pytorch/Outputs/evaluations/frcnn-R-50-C4-1x/cs6/train-CS6-Train-HP-5k_val-easy_conf-0.1/eval-dets_val_easy'
cs6_hp_5k_model_det_json = '/mnt/nfs/work1/elm/pchakrabarty/cs6_jsons/train-CS6-Train-HP-5k_val-easy_conf-0.1_cs6_annot_eval_scores.json'

# Select fddb and json files
fddb_file = cs6_hp_and_wider_bs512_gpu4_5k_model_det_fddb
json_file = cs6_hp_and_wider_bs512_gpu4_5k_model_det_json

def convert_to_json(fddb_file):
    ds_json = {'images' : [],
               'categories' : [{'id':1,'name':'person'}],
               'annotations' : []
              }

    # If a dir, concatenate text files in it: all videos
    if os.path.isdir(fddb_file):
        txt_files = [t for t in os.listdir(fddb_file) if t.endswith('.txt')]
        all_det = ''
        for t in txt_files:
            with open(os.path.join(fddb_file,t),'r') as f:
                all_det += f.read()
            f.close()
        with tempfile.NamedTemporaryFile(mode='w',delete=True) as tmp:
            tmp.write(all_det)
            ann = parse_wider_gt(tmp.name)
    else:
        # Parse fddb file to a dict
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
        #print(bboxes)
        for bbox in bboxes:
            bbox[:4] = map(int,bbox[:4])
            bbox_prop = {'id':bbox_id,
                         'image_id':fid,
                         'segmentation':[],
                         'category_id':1,
                         'iscrowd':0,
                         'bbox':bbox,
                         'area':bbox[2]*bbox[3]
                        }
            #print(bbox_prop['bbox'],'--',bbox_prop['area'])
            ds_json['annotations'].append(bbox_prop)
            bbox_id += 1
    print('Total:',fid,'files saved in JSON format')
    with open(json_file,'w') as f:
        json.dump(ds_json,f)
    f.close()

if __name__ == '__main__':
    convert_to_json(fddb_file)


