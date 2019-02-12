"""
Takes a JSON file and output annotation for only one video.
srun --pty --mem 10000 python tools/face/mod_json.py
=======

1) Takes a JSON file and output annotation for only one video.
2) Make compatible with distillation loss
3) Add dataset name into annotations
4) Add artificial noise to the labels
5) Remove all tracklet bboxes from HP JSON [detector+tracker]

srun --pty --mem 10000 python tools/face/mod_json.py


By default output JSONs are saved under 'Outputs/modified_annots/'


Usage 1: add "dataset" field to each annot
------------------------------------------
srun --pty --mem 10000 python tools/face/mod_json.py \
    --task dataset-annot \
    --dataset_name cs6-train-hp \
    --json_file data/CS6_annot/cs6-train-hp.json


Usage 2: add noise to the annotations
------------------------------------------
srun --pty --mem 10000 python tools/face/mod_json.py \
    --task noisy-label \
    --bbox_noise_level 1.0 \
    --dataset_name cs6-train-hp \
    --json_file data/CS6_annot/cs6-train-hp.json


Usage 3: add "dataset" and "source" fields
------------------------------------------
srun --pty --mem 10000 python tools/face/mod_json.py \
    --task dataset-annot \
    --dataset_name cs6-train-hp \
    --add_source \
    --json_file data/CS6_annot/cs6-train-hp.json


Usage 4: remove bboxes that come from tracker
---------------------------------------------
srun --pty --mem 10000 python tools/face/mod_json.py \
    --task only-dets \
    --dataset_name cs6-train-hp \
    --json_file data/CS6_annot/cs6-train-hp.json


Usage 5: BDD Subsample images with a given constraint
---------------------------------------------
srun --pty --mem 10000 python tools/face/mod_json.py \
    --task subsample \
    --bdd_constraints <weather> <scene> <timeofday> \
    --dataset_name bdd_HP18k \
    --json_file data/bdd_jsons/bdd_HP18k.json

Usage 6: Subsample annotations with a confidence score higher than a threshold
---------------------------------------------
srun --pty --mem 10000 python lib/datasets/bdd100k/mod_json.py \
     --task thresh-score \
     --dataset_name bdd_HP18k \
     --json_file data/bdd_jsons/bdd_HP18k.json


Usage 7: Randomly Subsample a given fraction of images
---------------------------------------------
srun --pty --mem 10000 python lib/datasets/bdd100k/mod_json.py \
     --task random-subsample \
     --dataset_name bdd_peds_not_clear_any_daytime_train_100 \
     --json_file data/bdd_jsons/vary_pure_target/bdd_peds_not_clear_any_daytime_train.json \
     --fraction 1.0
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
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
from matplotlib import pyplot as plt


JSON_FILE = 'data/CS6_annot/cs6-train-gt_noisy-0.5.json'
# OUT_DIR = '/mnt/nfs/work1/elm/arunirc/Data/CS6_annots'
OUT_DIR = 'Outputs/modified_annots/'
VID_NAME = '3013.mp4'
DEBUG = True
NOISE_LEVEL = 0.5

JSON_FILE_SCORES = 'data/CS6_annot/cs6-train-det-score_face_train_annot_coco_style.json'

BDD_CONSTRAINTS = ['any','any','any']
BDD_DIR = 'data/bdd100k/'
BDD_VID_DIR = 'data/bdd_peds_HP18k'
BDD_HP_JSON = 'data/bdd_jsons/bdd_HP18k.json'
BDD_TARGET_TRAIN_JSON = 'data/bdd_jsons/bdd_not_clear_any_daytime_train.json'

args = None

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
    parser.add_argument(
        '--add_source', help='Source field annotation', 
        action='store_true'
    )
    parser.add_argument(
        '--bdd_constraints', help='List of constraints to subsample a BDD json',
        default=BDD_CONSTRAINTS, nargs=3
    )
    parser.add_argument(
        '--bdd_hp_json_file', help='Json file for BDD HP',
        default=BDD_HP_JSON
    )
    parser.add_argument(
        '--bdd_train_json_file', help='Json file for BDD train',
        default=BDD_TARGET_TRAIN_JSON
    )
    parser.add_argument(
        '--map_fn', help='Function to map scores to new values',
        default='random'
    )
    parser.add_argument(
        '--thresh', help='Threshold: keep scores above this value',
        default=0.5, type=float
    )
    parser.add_argument(
        '--lmbda', help='Lambda for distill softening',
        default=1.0, type=float
    )
    parser.add_argument(
        '--fraction', help='Fraction of images to subsample',
        default=1.0, type=float
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
            outfile.write(json.dumps(ann_dict, indent=2))


# ------------------------------------------------------------------------------
def make_noisy_annots(output_dir, json_file, bbox_noise_level=0.3, 
                      img_noise_level=1.0):
# ------------------------------------------------------------------------------
    '''
        Add noise to the bounding-box annotations of CS6 videos. This should 
        be trivially transferrable to any MS-COCO format JSON.
        A fraction of total images (img_noise_level*num_images) are selected to 
        have noise. By default, *all* images are selected.
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
            
            if annot_bbox[2] >= im_info['width']:
                annot_bbox[2] = im_info['width'] - 2
            if annot_bbox[3] >= im_info['height']:
                annot_bbox[3] = im_info['height'] - 2
            
            rand_x = float(np.random.randint(1, im_info['width'] - annot_bbox[2]) )
            rand_y = float(np.random.randint(1, im_info['height'] - annot_bbox[3]) )
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
            outfile.write(json.dumps(ann_dict, indent=2))



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

            #print(len(gt_bboxes))

    print(im_info)

    out_file = osp.join(output_dir,
               osp.splitext(osp.basename(json_noisy))[0]) \
               + '_distill.json'
    with open(out_file, 'w', encoding='utf8') as outfile:
        outfile.write(json.dumps(ann_noisy_dict, indent=2))


# ------------------------------------------------------------------------------
def add_dataset_annots(output_dir, json_file, data_set):
# ------------------------------------------------------------------------------
    ''' Add a 'dataset' field to every annotation '''
    if not osp.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(json_file) as f:
        ann_dict = json.load(f)
    print(ann_dict.keys())

    #print(len(gt_bboxes))
    #print(im_info)
    # ann_dict['images'] = vid_images
    for annot in ann_dict['annotations']:
        annot['dataset'] = data_set
        if args.add_source:
            # default: assume source is detector (1) not tracker (2)
            annot['source'] = 1 


    out_file = osp.join(output_dir, 
                osp.splitext(osp.basename(json_file))[0]) \
                + '_dataset-' + data_set + '.json'
    with open(out_file, 'w', encoding='utf8') as out:
        out.write(json.dumps(ann_dict, indent=2))



# ------------------------------------------------------------------------------
def remove_tracklet_annots(output_dir, json_file):
# ------------------------------------------------------------------------------
    ''' Remove HP JSON annotations that come from tracklets '''
    if not osp.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(json_file) as f:
        ann_dict = json.load(f)
    print(ann_dict.keys())

    det_annots = [x for x in ann_dict['annotations'] if x['source'] == 1]
    ann_dict['annotations'] = det_annots
    out_file = osp.join(output_dir, 
                osp.splitext(osp.basename(json_file))[0]) \
                + '-det' + '.json'

    print('Output: ' + out_file)
    with open(out_file, 'w', encoding='utf8') as outfile:
        outfile.write(json.dumps(ann_dict, indent=2))


# ------------------------------------------------------------------------------
def subsample_bdd(output_dir, json_file, constraints):
# ------------------------------------------------------------------------------
    ''' Subsample images from the BDD json satisfying special constraint on the weather,  and time of day '''
    if not osp.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print('Loading json file...')
    with open(json_file) as f:
        ann_dict = json.load(f,object_pairs_hook=collections.OrderedDict)
    print(ann_dict.keys())
    print('...done')

    # Get samples -- video names from the json
    lab_files = list(set([i['file_name'].strip().split('_')[0].split('.')[0]+'.json' for i in ann_dict['images']]))

    out_file = osp.join(output_dir,
                osp.splitext(osp.basename(json_file))[0]) \
                + '_'+('_'.join(constraints)) + '.json'
    
    # Get list of videos satisfying the constraints
    vid_list = []
    ann_dir = os.path.join(BDD_DIR,'labels','100k','val')
    sel_attrib = {
        'weather'   : ([] if constraints[0].strip() == 'any' else constraints[0].strip().split(',') ), #[constraints[0]]),
        'scene'     : ([] if constraints[1].strip() == 'any' else constraints[1].strip().split(',') ), #[constraints[1]]),
        'timeofday' : ([] if constraints[2].strip() == 'any' else constraints[2].strip().split(',') ), #[constraints[2]])
    }
    print('Subsampling',json_file,'to keep images with this constraint:',sel_attrib)
    
    for lab_file in lab_files:
        lab_file = osp.split(lab_file)[-1]
        lab_file_path = osp.join(ann_dir,lab_file)
        with open(lab_file_path,'r') as f:
            data = json.load(f,object_pairs_hook=collections.OrderedDict)
        name = data['name']
        attrib = data['attributes']
        # check allowed conditions
        allowed = True
        for attr in ['weather','scene','timeofday']:
            if len(sel_attrib[attr]) > 0:
                if not (attrib[attr] in sel_attrib[attr]):
                    allowed = False
                    break
        if not allowed:
            continue
        ##################
        vid_list.append(lab_file.split('.')[0])
    ####################
    print('Number of videos in subsampled JSON:',len(vid_list))
    
    # map image id to video
    img_list = ann_dict['images']
    img_id2vid_name = {img['id'] : osp.split(img['file_name'])[-1].split('_')[0].split('.')[0] for img in img_list}
    
    # make new ann dict
    new_ann_dict = {}
    new_ann_dict['annotations'] = [ann for ann in ann_dict['annotations'] if img_id2vid_name[ann['image_id']] in vid_list]
    new_ann_dict['categories'] = ann_dict['categories']
    new_ann_dict['images'] = [img for img in img_list if osp.split(img['file_name'])[-1].split('_')[0].split('.')[0] in vid_list]
    
    new_ann_dict = collections.OrderedDict(new_ann_dict)
       
    print('Number of images:',len(new_ann_dict['images']))
    print('Number of annotations:',len(new_ann_dict['annotations']))

    print('Output: ' + out_file)
    with open(out_file, 'w', encoding='utf8') as outfile:
        outfile.write(json.dumps(new_ann_dict, indent=2))



# ------------------------------------------------------------------------------
def make_bdd_test(output_dir, bdd_train_json_file, bdd_hp_json_file):
# ------------------------------------------------------------------------------
    ''' Keep images from the BDD json not in the training set '''
    with open(bdd_hp_json_file) as f:
        hp_dict = json.load(f,object_pairs_hook=collections.OrderedDict)
    print(hp_dict.keys())    
    with open(bdd_train_json_file) as f:
        train_dict = json.load(f,object_pairs_hook=collections.OrderedDict)
    print(train_dict.keys())


    def fname2vidname(t):
        return osp.split(t)[-1].split('.')[0].split('_')[0]
    img_id2vidname = {img['id'] : fname2vidname(img['file_name']) for img in train_dict['images']}
    img_id2path = {img['id'] : img['file_name'] for img in train_dict['images']}

    hp_vid_list = set([fname2vidname(img['file_name']) for img in hp_dict['images']])
    train_vid_list = set([fname2vidname(img['file_name']) for img in train_dict['images']])
    leftover_vid_list = list(train_vid_list-hp_vid_list)

    print(len(train_vid_list))
    print(len(hp_vid_list))
    print(len(leftover_vid_list))

    new_ann_dict = {}
    new_ann_dict['categories'] = hp_dict['categories']
    new_ann_dict['images'] = [img for img in train_dict['images'] if \
                                ((fname2vidname(img['file_name']) in leftover_vid_list) and (osp.exists(img['file_name'])))]
    new_ann_dict['annotations'] = [ann for ann in train_dict['annotations'] if \
                                    ((img_id2vidname[ann['image_id']] in leftover_vid_list) and (osp.exists(img_id2path[ann['image_id']])))]

    out_file = 'bdd_test.json'
    print('Output: ' + out_file)
    with open(osp.join(output_dir,out_file), 'w', encoding='utf8') as outfile:
        outfile.write(json.dumps(new_ann_dict, indent=2))


# ------------------------------------------------------------------------------
def map_scores(output_dir, json_file, lmbda=1.0, map_fn='random'):
# ------------------------------------------------------------------------------
    with open(json_file,'r') as f:
        ann_dict = json.load(f)
    f.close()

    map_fn = map_fn.strip()

    if not (map_fn in ['random','distill']):
        if os.path.exists(map_fn):
            score_map = np.load(map_fn)
            scores_binned,scores_matched = score_map
            order = np.argsort(scores_binned)
            
            out_file = osp.join(output_dir,
                        osp.splitext(osp.basename(json_file))[0]+'_remapped_hist.json')
        else:
            print('ERROR: no file',map_fn)
            return
    elif map_fn == 'random':
        out_file = osp.join(output_dir,
                    osp.splitext(osp.basename(json_file))[0]+'_remapped_random.json')
    elif map_fn == 'distill':
        out_file = osp.join(output_dir,
                    osp.splitext(osp.basename(json_file))[0]+'_remapped_distill-'+str(lmbda)+'.json')

    for ann in ann_dict['annotations']:
        if ann['source'] == 1.0:
            if map_fn == 'random':
                ann['dataset'] = ann['dataset']+'_remap_random'
                ann['score'] = np.random.uniform(0.81, 1)
            elif map_fn == 'distill':
                ann['dataset'] = ann['dataset']+'_remap_distill-'+str(lmbda)
                ann['score'] = lmbda * ann['score'] + (1-lmbda) * 1
            else:
                ann['dataset'] = ann['dataset']+'_remap_hist'
                ann['score'] = np.interp(ann['score'],
                                scores_binned[order],
                                scores_matched[order])

    print('Output: ' + out_file)
    with open(out_file, 'w', encoding='utf8') as outfile:
        outfile.write(json.dumps(ann_dict, indent=2))
    outfile.close()
    print('JSON with mapped scores saved to:',out_file)


# ------------------------------------------------------------------------------
def thresh_score(output_dir, json_file, thresh):
# ------------------------------------------------------------------------------
    with open(json_file,'r') as f:
        ann_dict = json.load(f)
    f.close()
    
    ann_dict['annotations'] = [ann for ann in ann_dict['annotations'] if ((ann['source'] == 1.0) and (ann['score'] >= thresh))]
    ann_dict = collections.OrderedDict(ann_dict)

    out_file = osp.join(output_dir,osp.splitext(osp.basename(json_file))[0]+'_thresh-'+str(thresh)+'.json')
    print('New json file saved to: ' + out_file)
    with open(out_file, 'w', encoding='utf8') as outfile:
        outfile.write(json.dumps(ann_dict, indent=2))
    outfile.close()
    print('JSON with mapped scores saved to:',out_file)


# ------------------------------------------------------------------------------
def random_subsample(output_dir, json_file, fraction, data_set):
# ------------------------------------------------------------------------------
    ''' Randomly subsample a given fraction of the images from a JSON '''
    if not osp.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print('Loading json file...')
    with open(json_file) as f:
        ann_dict = json.load(f,object_pairs_hook=collections.OrderedDict)
    print(ann_dict.keys())
    print('...done')

    new_ann_dict = {}
    new_ann_dict['categories'] = ann_dict['categories']
    
    img_list = ann_dict['images']
    
    # use if some images are missing from the images folder
    #img_list = [im for im in img_list if os.path.exists(im['file_name'])]
    
    num_img_frac = int(fraction*len(img_list))
    np.random.shuffle(img_list)
    new_ann_dict['images'] = img_list[:num_img_frac]
    new_img_id_list = [im['id'] for im in new_ann_dict['images']]

    new_ann_dict['annotations'] = [ann for ann in ann_dict['annotations'] if ann['image_id'] in new_img_id_list]
    if not data_set is None:
        for annot in ann_dict['annotations']:
            annot['dataset'] = data_set

    new_ann_dict = collections.OrderedDict(new_ann_dict)
    print('Number of images:',len(new_ann_dict['images']))
    print('Number of annotations:',len(new_ann_dict['annotations']))
    out_file = osp.join(output_dir,
                osp.splitext(osp.basename(json_file))[0]) \
                + '_subsampled_'+ str(fraction) + '.json'
    print('Output: ' + out_file) 
    with open(out_file, 'w', encoding='utf8') as outfile:
        outfile.write(json.dumps(new_ann_dict, indent=2))


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

    elif args.task == 'only-dets':
       remove_tracklet_annots(output_dir, json_file)
        
    elif args.task == 'subsample-bdd':
        np.random.seed(0)
        subsample_bdd(output_dir, json_file, args.bdd_constraints)

    elif args.task == 'make-bdd-test':
        np.random.seed(0)
        bdd_train_json_file = args.bdd_train_json_file
        bdd_hp_json_file = args.bdd_hp_json_file
        make_bdd_test(output_dir, json_file, bdd_hp_json_file)
    
    elif args.task == 'map-scores':
        np.random.seed(0)
        map_scores(output_dir, json_file, lmbda=args.lmbda, map_fn=args.map_fn)

    elif args.task == 'thresh-score':
        np.random.seed(0)
        thresh_score(output_dir, json_file, args.thresh)

    elif args.task == 'random-subsample':
        np.random.seed(0)
        random_subsample(output_dir, json_file, args.fraction, args.dataset_name)

    else:
        raise NotImplementedError
