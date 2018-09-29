
"""

Takes a JSON file and output annotation for only one video.

srun --pty --mem 10000 python tools/face/mod_json.py


By default output JSONs are saved under 'Outputs/modified_annots/'


Usage 1: add "dataset" field to each annot
------------------------------------------
srun --pty --mem 10000 python tools/face/mod_json.py \
    --task dataset-annot \
    --dataset_name cs6-train-hp \
    --json_file data/CS6_annot/cs6-train-hp.json


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import sys
sys.path.append('./tools')
import _init_paths
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
from utils import face_utils




JSON_FILE = 'data/CS6_annot/cs6-train-gt_noisy-0.5.json'
# OUT_DIR = '/mnt/nfs/work1/elm/arunirc/Data/CS6_annots'
OUT_DIR = 'Outputs/modified_annots/'
VID_NAME = '3013.mp4'
DEBUG = True
NOISE_LEVEL = 0.5

JSON_FILE_SCORES = 'data/CS6_annot/cs6-train-det-score_face_train_annot_coco_style.json'

def parse_args():
    parser = argparse.ArgumentParser(description='Modify CS6 ground truth data')
    parser.add_argument(
        '--task', help='Specify the modification', default='distill-label'
    )
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
        default='data/CS6_annot'
    )
    parser.add_argument(
        '--bbox_noise_level',
        default=NOISE_LEVEL, type=float
    )
    parser.add_argument(
        '--json_file_scores', help='Name of scores JSON file', 
        default=JSON_FILE_SCORES
    )
    parser.add_argument(
        '--dataset_name', help='Name of dataset', 
        default=None
    )
    return parser.parse_args()



# ------------------------------------------------------------------------------
def single_video_annots(output_dir, video_file, json_file):
# ------------------------------------------------------------------------------
    ''' replace the images and annotations with only those from specified video '''
    if not osp.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        video_name = osp.splitext(video_file)[0] # strip extension

        with open(json_file) as f:
            ann_dict = json.load(f)
        print(ann_dict.keys())

        vid_images = [x for x in ann_dict['images'] \
                        if video_name+'_' in x['file_name']]
        vid_image_ids = set([x['id'] for x in vid_images])
        vid_annots = [x for x in ann_dict['annotations'] if x['image_id'] in \
                        vid_image_ids]

        ann_dict['images'] = vid_images
        ann_dict['annotations'] = vid_annots
        out_file = osp.join(output_dir, 
                    osp.splitext(osp.basename(json_file))[0]) \
                    + '_' + video_name + '.json'
        with open(out_file, 'w', encoding='utf8') as outfile:
            outfile.write(json.dumps(ann_dict))




# ------------------------------------------------------------------------------
def make_noisy_annots(output_dir, json_file, bbox_noise_level=0.3, 
                      img_noise_level=1.0):
# ------------------------------------------------------------------------------
    '''
        Add noise to the bounding-box annotations of CS6 videos. This should 
        be trivially transferrable to any MS-COCO format JSON.

        A fraction of total images (img_noise_level*num_images) are selected to 
        have noise.
        In each image, max(1, bbox_noise_level*num_bboxes) bounding boxes  
        are selected to have their X and Y se to a random position in the image.
    '''
    if not osp.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(json_file) as f:
            ann_dict = json.load(f)
    if DEBUG:
        print(ann_dict.keys())

    num_total_img = len(ann_dict['images'])
    num_sel_img = int(img_noise_level * num_total_img)
    perm_ann_dict = np.random.permutation(ann_dict['images'])
    images_sel = perm_ann_dict[0:num_sel_img]
    image_ids = [x['id'] for x in images_sel]
    vid_annots = []

    for (i,(im_id, im_info)) in enumerate(zip(image_ids, images_sel)):
        annots = [x for x in ann_dict['annotations'] if x['image_id'] == im_id]
        # print(im_id)

        # pick a random subset of annots to perturb
        num_total_annot = len(annots)
        num_sel_noisy = int(max(1, bbox_noise_level * num_total_annot))
        annots_sel = np.random.permutation(annots)

        for j in range(num_sel_noisy):
            # move some bboxes to random locations (size remains same)
            annot_bbox = annots_sel[j]['bbox']
            rand_x = float(np.random.randint(1, im_info['width'] - annot_bbox[2]))
            rand_y = float(np.random.randint(1, im_info['height'] - annot_bbox[3]))
            annot_bbox[0] = rand_x # Modifications propagated through Call-by-Object
            annot_bbox[1] = rand_y # annot_bbox --> annots --> ann_dict['annotations']

        # print(im_id)
        if ((i + 1) % 100) == 0:
            sys.stdout.write('(%d/%d) ' % (i, num_sel_img))
            sys.stdout.flush()

    # Writing modified JSON
    out_file = osp.join(output_dir, 
                    osp.splitext(osp.basename(json_file))[0]) \
                    + ('_noisy-%.2f.json' % bbox_noise_level)
    with open(out_file, 'w', encoding='utf8') as outfile:
            outfile.write(json.dumps(ann_dict))





# ------------------------------------------------------------------------------
def make_distill_annots(output_dir, json_noisy, json_dets):
# ------------------------------------------------------------------------------
    ''' 
        Combine detection scores annotations JSON and noisy labels annotations 
        JSON files to create a single JSON annotations file for use in training 
        with a distillation loss 
    '''

    # TODO  .... 

    if not osp.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(json_noisy) as f:
        ann_noisy_dict = json.load(f)

    with open(json_dets) as g:
        ann_det_dict = json.load(g)

    if DEBUG:
        print(ann_noisy_dict.keys())

    # map 
    det_image_names = {x['file_name']: x['id'] for x in ann_det_dict['images'] \
                            if '1101_' not in x['file_name']}
    gt_images = [x for x in ann_noisy_dict['images'] \
                    if '1101_' not in x['file_name']]

    for im_info in gt_images:
        # ground-truth annot
        im_id = im_info['id']
        gt_annots = [x for x in ann_noisy_dict['annotations'] \
                            if x['image_id'] == im_id]

        # locate that image in detections annot
        det_im_id = det_image_names.get(im_info['file_name'])

        if det_im_id == None:
            # the gt image has no corresponding image in detections annots
            for gt_ann in gt_annots:
                gt_ann['score'] = 0.0
        else:
            gt_bboxes = [x['bbox'] for x in gt_annots]
            det_bboxes = [x['bbox'] for x in ann_det_dict['annotations'] \
                            if x['image_id'] == det_im_id]


            # bipartite matching (Hungarian algorithm)
            gt_bboxes = np.array(gt_bboxes)
            det_bboxes = np.array(det_bboxes)
            # convert [x1 y1 w h] to [x1 y1 x2 y2]
            gt_bboxes[:,2] += gt_bboxes[:,0]
            gt_bboxes[:,3] += gt_bboxes[:,1]
            det_bboxes[:,2] += det_bboxes[:,0]
            det_bboxes[:,3] += det_bboxes[:,1]

            idx_gt, idx_pred, iou_mat,_ = face_utils.match_bboxes(
                                            gt_bboxes, det_bboxes)

            if len(idx_gt) > 0 and len(idx_pred) > 0:
                print(iou_mat)
            else:
                for gt_ann in gt_annots:
                    gt_ann['score'] = 0.0

            print(len(gt_bboxes))


        print(im_info)
        # TODO - sanity-check: assert all gt-annotations have a key 'score'

    # ann_dict['images'] = vid_images
    # ann_dict['annotations'] = vid_annots

    out_file = osp.join(output_dir, 
                osp.splitext(osp.basename(json_noisy))[0]) \
                + '_distill.json'
    with open(out_file, 'w', encoding='utf8') as outfile:
        outfile.write(json.dumps(ann_noisy_dict))




# ------------------------------------------------------------------------------
def add_dataset_annots(output_dir, json_file, data_set):
# ------------------------------------------------------------------------------
    ''' Add a 'dataset' field to every annotation '''
    if not osp.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(json_file) as f:
        ann_dict = json.load(f)
    print(ann_dict.keys())

    # ann_dict['images'] = vid_images
    for annot in ann_dict['annotations']:
        annot['dataset'] = data_set

    out_file = osp.join(output_dir, 
                osp.splitext(osp.basename(json_file))[0]) \
                + '_dataset-' + data_set + '.json'
    with open(out_file, 'w', encoding='utf8') as out:
        out.write(json.dumps(ann_dict, indent=2))



if __name__ == '__main__':
    

    args = parse_args()
    print(args)
    output_dir = args.output_dir
    json_file = args.json_file


    if args.task == 'single-video':
        video_file == args.video
        single_video_annots(output_dir, video_file, json_file)
       
    elif args.task == 'noisy-label':
        np.random.seed(0)
        make_noisy_annots(output_dir, json_file, 
                          bbox_noise_level=args.bbox_noise_level)

    elif args.task == 'distill-label':
        # TODO
        np.random.seed(0)
        make_distill_annots(output_dir, json_file, args.json_file_scores)

    elif args.task == 'dataset-annot':
        assert(args.dataset_name is not None)
        add_dataset_annots(output_dir, json_file, args.dataset_name)
        
    else:
        raise NotImplementedError

    
    





    

        
        
        
        
