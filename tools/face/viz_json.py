
"""

Takes a JSON file and visualizes the annotation boxes on images.

srun --mem 10000 python tools/face/viz_json.py

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



JSON_FILE = 'data/CS6_annot/cs6-train-easy-gt-sub.json'
# OUT_DIR = '/mnt/nfs/work1/elm/arunirc/Data/CS6_annots'
OUT_DIR = 'Outputs/visualizations/'

DEBUG = False

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

    with open(args.json_file) as f:
        ann_dict = json.load(f)
    print(ann_dict.keys())

    out_dir = osp.join(args.output_dir, 
                       osp.splitext(osp.basename(args.json_file))[0])
    if not osp.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)    
    
    i = 0
    for img_annot in tqdm(ann_dict['images']):
        image_name = img_annot['file_name']
        image_id = img_annot['id']
        bboxes = [x['bbox'] for x in ann_dict['annotations'] \
                        if x['image_id'] == image_id]
        im = cv2.imread(osp.join(args.imdir, image_name))
        assert im.size > 0
        im_det = draw_detection_list(im, np.array(bboxes))
        out_path = osp.join(out_dir, image_name.replace('/', '_'))
        cv2.imwrite(out_path, im_det)
        i += 1 
        if i == 5000:
            break





    

        
        
        
        
