
"""

Takes a JSON file and output annotation for only one video.

srun --pty --mem 20000 python tools/face/mod_json_subsample.py

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
sys.path.append('./tools')
import numpy as np
import os, cv2
import argparse
import os.path as osp
import time
import skvideo.io
import json
import csv
from six.moves import xrange
from PIL import Image
from tqdm import tqdm



JSON_FILE = 'data/CS6_annot/cs6-train-gt_face_train_annot_coco_style.json'
# OUT_DIR = '/mnt/nfs/work1/elm/arunirc/Data/CS6_annots'
OUT_DIR = 'Outputs/modified_annots/'


def parse_args():
    parser = argparse.ArgumentParser(description='Creating CS6 ground truth data')
    parser.add_argument(
        '--output_dir', help='directory for saving outputs',
        default=OUT_DIR, type=str
    )
    parser.add_argument(
        '--json_file', help='Name of JSON file', default=JSON_FILE
    )
    parser.add_argument(
        '--ratio', default=0.5, type=float
    )
    parser.add_argument(
        '--imdir', help="root directory for loading dataset images",
        default='data/CS6_annot', type=str)
    return parser.parse_args()







if __name__ == '__main__':
    

    args = parse_args()

    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    with open(args.json_file) as f:
        ann_dict = json.load(f)
    print(ann_dict.keys())

    
    annots = [x for x in ann_dict['annotations']]
    np.random.shuffle(annots)
    annots_subset = annots[0: int(args.ratio * len(annots))]
    
    ann_dict['annotations'] = annots_subset
    out_file = osp.join(args.output_dir, 
                osp.splitext(osp.basename(args.json_file))[0]) + \
                '_subset-%.2f.json' % args.ratio
    with open(out_file, 'w', encoding='utf8') as outfile:
        outfile.write(json.dumps(ann_dict))

    
    





    

        
        
        
        
