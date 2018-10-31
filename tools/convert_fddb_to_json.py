import re
import os
import sys
import cv2
import json
import time
import random
import pickle
import imutils
import tempfile
import subprocess
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

# BDD pedestrian detections
bdd_peds_dets_20k_conf050_fddb = '/mnt/nfs/scratch1/ashishsingh/FALL2018/Detectron-pytorch-video/Outputs/detections_videos/frcnn-R-50-C4-1x/bdd20k/train-BDD_PED_NEW_val-video_conf-0.50/person'
bdd_peds_dets_conf050_json = 'test_det/bdd_peds_clear_any_daytime_det_conf050.json' 

bdd_peds_dets_conf070_json = 'test_det/bdd_peds_clear_any_daytime_det_conf070.json'

bdd_peds_dets_conf080_json = 'test_det/bdd_peds_clear_any_daytime_det_conf080.json'       

bdd_peds_dets_conf090_json = 'test_det/bdd_peds_clear_any_daytime_det_conf090.json'       

# Select fddb and json files
fddb_file = bdd_peds_dets_20k_conf050_fddb
json_file = 'tmp/conf080_det.json' #bdd_peds_dets_conf080_json

conf_thresh = 0.8

n_samples = 100000

#det_vid_dir = ''
#save_det_dir = ''
det_vid_dir = '/mnt/nfs/scratch1/ashishsingh/FALL2018/BDD20k/BDD20k'
#save_det_dir = 'data/bdd_detections_20k'
save_det_dir = '/mnt/nfs/scratch1/pchakrabarty/bdd_detections_20k_fast'


def _ffmpeg_extract_rotation(pathtofile):
    # https://stackoverflow.com/a/14237677/297353  
    cmd = 'ffmpeg -i %s' % pathtofile
    p = subprocess.Popen(
        cmd.split(" "),
        stderr = subprocess.PIPE,
        close_fds=True
    )
    stdout, stderr = p.communicate()
    reo_rotation = re.compile('rotate\s+:\s(?P<rotation>.{4})')
    stderr = str(stderr)
    match_rotation = reo_rotation.search(stderr)
    if match_rotation is None:
        input(stderr)
        input(pathtofile)
        return 0
    match = match_rotation.groups()[0].strip()
    match = match.replace('\\','')
    match = match.replace('n','')
    rotation = float(match)
    return rotation


def load_vid(vid_file,rotation=None):
    vid = []
    if rotation is None:
        rotation = _ffmpeg_extract_rotation(vid_file)
    videogen = cv2.VideoCapture(vid_file)
    while True:
        ret,im = videogen.read()
        if not ret:
            break
        im = imutils.rotate_bound(im,rotation-360)
        vid.append(im)
    videogen.release()
    return vid

def bin_video_files(all_file_list):
    vid_map = {}
    for fl in all_file_list:
        uind = fl.index('_')
        vname = fl[:uind]
        fid = fl[(uind+1):]
        if vname in vid_map:
            vid_map[vname].append(fid)
        else:
            vid_map[vname] = [fid]
    return vid_map

def save_vid_frames(vid_name,vid_file_path,frame_list,save_det_dir,rotation=None):
    vid = load_vid(vid_file_path,rotation=rotation)
    for frame_id_str in frame_list:
        frame_id = int(frame_id_str)
        cv2.imwrite(os.path.join(save_det_dir,vid_name+'_'+frame_id_str+'.jpg'),vid[frame_id])

def convert_to_json(fddb_file,format='eval'):
    ds_json = {'images' : [],
               'categories' : [{'id':1,'name':'person'}],
               'annotations' : []
              }

    # If a dir, concatenate text files in it: all videos
    if os.path.isdir(fddb_file):
        txt_files = [t for t in os.listdir(fddb_file) if t.endswith('.txt')][:15000]
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
    all_keys = list(ann.keys())
    
    # randomly sample 100k frames
    def has_dets(ann,fname,conf_thresh):
        if int(fname[(fname.index('_')+1):]) > 1100: # BAD hack: to handle incomplete videos
            return False
        if len(ann[fname]) == 0:
            return False
        scores = np.array([b[4] for b in ann[fname]])
        if np.sum(scores>=conf_thresh) == 0:
            return False
        return True
    #all_keys = [k for k in all_keys if len(ann[k]) > 0] # keep only frames with detections
    all_keys = [k for k in all_keys if has_dets(ann,k,conf_thresh)] # keep only frames with detections
    random.shuffle(all_keys)
    all_keys = all_keys[:n_samples]
    all_keys.sort()
    ##### end of sampling #####

    all_files = []
    for fid,filename in enumerate(all_keys):
        all_files.append(filename)
        w,h,c = (720,1280,3)
        im_prop = {'width':w,
                   'height':h,
                   'id':fid,
                   'file_name':filename+'.jpg'
                  }
        ds_json['images'].append(im_prop)
        bboxes = ann[filename]
        print('Reading:',filename)
        for bbox in bboxes:
            score = bbox[4]
            bbox[:4] = map(int,bbox[:4])
            if format == 'coco':
                bbox = bbox[:4] #throw away score for coco format. preserve to use for eval
            bbox_prop = {'id':bbox_id,
                         'image_id':fid,
                         'segmentation':[],
                         'category_id':1,
                         'iscrowd':0,
                         'bbox':bbox,
                         'area':bbox[2]*bbox[3],
                         'score':score,
                         'source':'detection',
                        }
            bbox_id += 1
            if score < conf_thresh:
                continue
            ds_json['annotations'].append(bbox_prop)
        del ann[filename]
    print('Number of annotations:',len(ds_json['annotations']))
    print('Total:',fid,'files saved in JSON format')

    # saving final json
    with open(json_file,'w') as f:
        json.dump(ds_json,f)
    f.close()

    if len(det_vid_dir) == 0:
        return
    
    # bin frames to corresponding videos
    print('Grouping video frames...')
    video_frames = bin_video_files(all_files)
    vid_list = list(video_frames.keys())
    print('Total of '+str(len(video_frames.items()))+' videos with '+str(np.sum([len(t) for _,t in video_frames.items()]))+' frames')

    # get list of rotations
    print('Making list of rotations for video files...')
    vid_path_list = [os.path.join(det_vid_dir,v+'.mov') for v in vid_list]
    rot_list = list(map(_ffmpeg_extract_rotation,vid_path_list))
    print('...Done. '+str(len(rot_list))+' rotations for '+str(len(vid_list))+' videos')

    # save metadata
    #print('Saving metadata...')
    #metadata = zip(vid_list,vid_path_list,rot_list)
    #with open('metadata.pkl')
    #print('...Done. Metadata saved to:',)

    # saving video frames
    for v,vid_name in enumerate(vid_list):
        vid_file_path = os.path.join(det_vid_dir,vid_name+'.mov')
        vid_sel_frames = video_frames[vid_name]
        save_vid_frames(vid_name,vid_file_path,vid_sel_frames,save_det_dir,rotation=rot_list[v])
    

if __name__ == '__main__':
    if sys.argv[1] == 'coco':
        convert_to_json(fddb_file,format='coco')
    else:
        convert_to_json(fddb_file,format='eval')


