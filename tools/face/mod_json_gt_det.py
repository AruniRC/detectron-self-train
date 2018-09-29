
"""

Subsample from CS6 GT only those images that are used in the Dets or HP JSON.


srun --mem 10000 python tools/face/mod_json_gt_det.py 

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



GT_JSON_FILE = 'data/CS6_annot/cs6-train-easy-gt.json'

HP_JSON_FILE = 'data/CS6_annot/cs6-train-easy-hp.json'

# OUT_DIR = '/mnt/nfs/work1/elm/arunirc/Data/CS6_annots'
OUT_DIR = 'Outputs/modified_annots/'

DEBUG = False

def parse_args():
    parser = argparse.ArgumentParser(description='Modifying CS6 ground truth data')
    parser.add_argument(
        '--output_dir', help='directory for saving outputs',
        default=OUT_DIR, type=str
    )
    parser.add_argument(
        '--gt_json_file', default=GT_JSON_FILE
    )
    parser.add_argument(
        '--hp_json_file', default=HP_JSON_FILE
    )
    parser.add_argument(
        '--imdir', help="root directory for loading dataset images",
        default='data/CS6_annot', type=str)
    return parser.parse_args()




_GREEN = (18, 127, 15)
color_dict = {'red': (0,0,225), 'green': (0,255,0), 'yellow': (0,255,255), 
                'blue': (255,0,0), '_GREEN':(18, 127, 15), '_GRAY': (218, 227, 218)}
# -----------------------------------------------------------------------------------
def draw_detection_list(im, dets):
# -----------------------------------------------------------------------------------
    """ Draw bounding boxes on a copy of image and return it.
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
        x0, y0, x1, y1 = [int(x) for x in bbox]
        line_color = color_dict['yellow']
        cv2.rectangle(im_det, (x0, y0), (x1, y1), line_color, thickness=2)

    return im_det






if __name__ == '__main__':
    

    args = parse_args()

    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # Load gt JSON
    with open(args.gt_json_file) as f:
        ann_dict = json.load(f)
    
    # Load HP JSON
    with open(args.hp_json_file) as f:
        hp_ann_dict = json.load(f)

    # Keep gt annots only for images in HP annots
    hp_images = set([ x['file_name'] for x in hp_ann_dict['images'] ])
    keep_images = [x for x in ann_dict['images'] if x['file_name'] in hp_images]
    keep_image_ids = set([x['id'] for x in keep_images])
    keep_annots = [x for x in ann_dict['annotations'] if x['image_id'] in keep_image_ids]

    # replace the images and annotations with only those from specified video
    ann_dict['images'] = keep_images
    ann_dict['annotations'] = keep_annots

    out_file = osp.join(args.output_dir, 
                osp.splitext(osp.basename(args.gt_json_file))[0]) + '-sub.json'

    with open(out_file, 'w', encoding='utf8') as outfile:
        outfile.write(json.dumps(ann_dict))

    
    





    

        
        
        
        
