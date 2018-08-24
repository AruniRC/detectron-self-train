
"""

Takes a JSON file and output annotation for only one video.

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



JSON_FILE = 'data/CS6_annot/cs6-subset-gt_face_train_annot_coco_style.json'
# OUT_DIR = '/mnt/nfs/work1/elm/arunirc/Data/CS6_annots'
OUT_DIR = 'Outputs/modified_annots/'
VID_NAME = '3013.mp4'
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
        '--video', help='Name of video file', default=VID_NAME
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

    video_file = args.video
    video_name = osp.splitext(video_file)[0] # strip extension

    with open(args.json_file) as f:
        ann_dict = json.load(f)
    print(ann_dict.keys())

    vid_images = [x for x in ann_dict['images'] \
                    if video_name+'_' in x['file_name']]
    vid_image_ids = set([x['id'] for x in vid_images])
    vid_annots = [x for x in ann_dict['annotations'] if x['image_id'] in vid_image_ids]

    # replace the images and annotations with only those from specified video
    ann_dict['images'] = vid_images
    ann_dict['annotations'] = vid_annots
    out_file = osp.join(args.output_dir, 
                osp.splitext(osp.basename(args.json_file))[0]) + '_' \
                + video_name + '.json'
    with open(out_file, 'w', encoding='utf8') as outfile:
        outfile.write(json.dumps(ann_dict))

    
    





    

        
        
        
        
