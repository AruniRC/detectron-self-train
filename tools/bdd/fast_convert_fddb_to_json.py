# Read the output of video mining: txt file with predictions in fddb format and convert to a json

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
import multiprocessing

import _init_paths
from utils.face_utils import parse_wider_gt
import core

dataset_root = '/mnt/nfs/work1/elm/arunirc/Research/detectron-video/mask-rcnn.pytorch/data/CS6_annot'

'''
## BDD Detections
fddb_file = '/mnt/nfs/scratch1/ashishsingh/FALL2018/Detectron-pytorch-video/Outputs/detections_videos/frcnn-R-50-C4-1x/bdd20k/train-BDD_PED_NEW_val-video_conf-0.50/person'
json_file = '/mnt/nfs/scratch1/pchakrabarty/trash/fast_conf080.json'
det_vid_dir = '/mnt/nfs/scratch1/ashishsingh/FALL2018/BDD20k/BDD20k'
save_det_dir = '/mnt/nfs/scratch1/pchakrabarty/bdd_detections_20k_fast/'
#################
'''
'''
## BDD Hard Positives
fddb_file = '/mnt/nfs/scratch1/ashishsingh/FALL2018/MDNet-tracklet/Output/bdd_hp/person/hp-res'
#json_file = '/mnt/nfs/scratch1/pchakrabarty/json_and_metadata_backup/bdd_HP.json'
json_file = '/mnt/nfs/scratch1/pchakrabarty/bdd_HP/bdd_HP.json'
det_vid_dir = '/mnt/nfs/scratch1/ashishsingh/FALL2018/BDD20k/BDD20k'
save_det_dir = '/mnt/nfs/scratch1/pchakrabarty/bdd_HP/'
#####################
'''
'''
## BDD Hard Positive -- 18k videos
fddb_file = '/mnt/nfs/scratch1/ashishsingh/FALL2018/MDNet-tracklet/Output/bdd_hp/person/hp-res'
json_file = '/mnt/nfs/scratch1/pchakrabarty/bdd_HP18k/bdd_HP18k.json'  #'./bdd_HP18k_thresh-050.json'
det_vid_dir = '/mnt/nfs/scratch1/ashishsingh/FALL2018/BDD20k/BDD20k'
save_det_dir = '/mnt/nfs/scratch1/pchakrabarty/bdd_HP18k/'
###################
'''
'''
## BDD Detections identical to HP
fddb_file = '/mnt/nfs/scratch1/ashishsingh/FALL2018/Detectron-pytorch-video/Outputs/detections_videos/frcnn-R-50-C4-1x/bdd20k/train-BDD_PED_NEW_val-video_conf-0.50/person'
json_file = '/mnt/nfs/scratch1/pchakrabarty/json_and_metadata_backup/bdd_dets_from_HP.json'
det_vid_dir = '/mnt/nfs/scratch1/ashishsingh/FALL2018/BDD20k/BDD20k'
save_det_dir = '/mnt/nfs/scratch1/pchakrabarty/bdd_HP18k/'
####################
'''
'''
## Cityscapes videos cars hard positives -- min tracklet len 5
fddb_file = '/mnt/nfs/scratch1/pchakrabarty/cityscapes/tracklet_len5'
json_file = 'cityscapes_cars_HPlen5.json'
det_vid_dir = '/mnt/nfs/scratch1/pchakrabarty/cityscapes/vid_dump/train'
save_det_dir = '/mnt/nfs/scratch1/pchakrabarty/cityscapes/cityscapes_cars_HP_frames'
####################
'''

## Cityscapes videos cars hard positives -- min tracklet len 3
fddb_file = '/mnt/nfs/scratch1/pchakrabarty/cityscapes/tracklet_len3'
json_file = 'cityscapes_cars_HPlen3.json'
det_vid_dir = '/mnt/nfs/scratch1/pchakrabarty/cityscapes/vid_dump/train'
save_det_dir = '/mnt/nfs/scratch1/pchakrabarty/cityscapes/cityscapes_cars_HP_frames'
####################

conf_thresh = 0.8 #0.5
num_samples = 100000
num_vid = 18000

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
    #print(stderr)
    #input('>>>'+str(rotation))
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
        vid.append(im)
    videogen.release()
    return vid

# get file name from path
def get_filename_from_path(fname):
    if len(os.path.split(fname)) > 1:
        return os.path.split(fname)[-1]
    return fname

def bin_video_files(all_file_list):
    vid_map = {}
    for fl in all_file_list:
        uind = fl.index('_')
        vname = fl[:uind]
        if len(os.path.split(vname)) > 1:
            vname = os.path.split(vname)[-1]
        if fl.endswith('.jpg'):
            fid = fl[(uind+1):-4] 
        else:   
            fid = fl[(uind+1):]
        if vname in vid_map:
            vid_map[vname].append(fid)
        else:
            vid_map[vname] = [fid]
        #input(vname+'--'+str(vid_map[vname]))
    return vid_map

def save_vid_frames(vid_name,vid_file_path,frame_list,save_det_dir,rotation=None):
    vid = load_vid(vid_file_path,rotation=rotation)
    for frame_id_str in frame_list:
        frame_id = int(frame_id_str)
        frame = imutils.rotate_bound(vid[frame_id],rotation-360)
        cv2.imwrite(os.path.join(save_det_dir,vid_name+'_'+frame_id_str+'.jpg'),frame)
    del(vid)

def convert_to_json(fddb_file,format='eval'):
    ds_json = {'images' : [],
               'categories' : [{'id':1,'name':'person'}],
               'annotations' : []
              }

    # If a dir, concatenate text files in it: all videos
    if os.path.isdir(fddb_file):
        txt_files = [t for t in sorted(os.listdir(fddb_file)) if t.endswith('.txt')]
        txt_files = txt_files[:num_vid]
        all_det = ''
        for t in txt_files:
            with open(os.path.join(fddb_file,t),'r') as f:
                all_det += f.read()
            f.close()
        
        tempfile.tempdir = '/mnt/nfs/scratch1/pchakrabarty/' # set temp dir to scratch1 if no space left here
        with tempfile.NamedTemporaryFile(mode='w',delete=True) as tmp:
            tmp.write(all_det)
            ann = parse_wider_gt(tmp.name)
    else:
        # Parse fddb file to a dict
        ann = parse_wider_gt(fddb_file)
    
    bbox_id = 0
    all_keys = list(ann.keys())
    
    def has_dets(ann,fname,conf_thresh):
        if fname.endswith('.jpg'):
            frame_id = int(fname[(fname.index('_')+1):-4])
        else:
            frame_id = int(fname[(fname.index('_')+1):])
        #if int(fname[(fname.index('_')+1):]) > 1100: # BAD hack: to handle incomplete videos
        if frame_id > 1000: # BAD hack: to handle incomplete videos
            return False
        if len(ann[fname]) == 0:
            return False
        scores = np.array([b[4] for b in ann[fname]])
        if np.sum(scores>=conf_thresh) == 0:
            return False
        return True
    
    # randomly sample 100k frames
    #all_keys = [k for k in all_keys if len(ann[k]) > 0] # keep only frames with detections
    #all_keys = [k for k in all_keys if has_dets(ann,k,conf_thresh)] # keep only frames with detections
    
    all_keys = [k for k in all_keys if has_dets(ann,k,conf_thresh)] # keep only frames with detections
    
    """
    # HACK to use the same frames as in a given HP json
    with open('data/bdd_jsons/bdd_HP18k.json','r') as f:
        d = json.load(f)
    f.close()
    all_keys = [os.path.split(img['file_name'])[-1] for img in d['images']]
    #print('>>>',all_keys[:10],ann.keys())
    """
    
    random.shuffle(all_keys)
    all_keys = all_keys[:num_samples]
    all_keys.sort()
    ##### end of sampling #####

    all_files = []
    for fid,filename in enumerate(all_keys):
        all_files.append(filename)
        w,h,c = (1024, 2048, 3) #(720,1280,3)

        if filename.endswith('.jpg'):
            im_file_name = filename
        else:
            im_file_name = filename+'.jpg'
        
        if len(os.path.split(im_file_name)) > 1:
            im_file_name = os.path.split(im_file_name)[-1]
        
        """
        filename = 'frames/'+im_file_name.split('_')[0].strip()+'/'+im_file_name #BAD hack
        if not filename in ann:
            continue
        """

        im_prop = {'width':w,
                   'height':h,
                   'id':fid,
                   #'file_name':filename+'.jpg'
                   'file_name':im_file_name
                  }
        ds_json['images'].append(im_prop)
        bboxes = ann[filename]
        print('Reading:',filename)
        for bbox in bboxes:
            #input(str(bbox))
            score = bbox[4]
            source = bbox[5]
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
                         'source':source,
                        }
            bbox_id += 1
            if score < conf_thresh:
                continue
            ds_json['annotations'].append(bbox_prop)
        del ann[filename]
    #ds_json['annotations'] = ds_json['annotations'][:num_samples]
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
    vid_path_list = [os.path.join(det_vid_dir,v+'.mov') for v in vid_list]
    
    print('Making list of rotations for video files...')
    rot_dict = dict(zip(vid_list,list(map(_ffmpeg_extract_rotation,vid_path_list))))
    print('...Done. '+str(len(rot_dict.keys()))+' rotations for '+str(len(vid_list))+' videos')

    # save meta data
    print('Saving video metadata...')
    metadata_file = os.path.join(save_det_dir,'metadata.pkl')
    with open(metadata_file,'wb') as f:
        pickle.dump([vid_list,video_frames,rot_dict],f)
    f.close()
    print('..Done. Saved to:',metadata_file)

def save_vid(det_dir,vid_range=(-1,-1)):
    with open(os.path.join(det_dir,'metadata.pkl'),'rb') as f:
        md = pickle.load(f)
    f.close()
    vid_list,video_frames,rot_dict = md
    # saving video frames
    if vid_range[0] == -1:
        for v,vid_name in enumerate(vid_list):
            vid_file_path = os.path.join(det_vid_dir,vid_name+'.mov')
            vid_sel_frames = video_frames[vid_name]
            save_vid_frames(vid_name,vid_file_path,vid_sel_frames,save_det_dir,rotation=rot_dict[vid_name])
    else:
        for v,vid_name in enumerate(vid_list[vid_range[0]:vid_range[1]]):
            #input('>>>'+str(v))
            vid_file_path = os.path.join(det_vid_dir,vid_name+'.mov')
            vid_sel_frames = video_frames[vid_name]
            save_vid_frames(vid_name,vid_file_path,vid_sel_frames,save_det_dir,rotation=rot_dict[vid_name])


if __name__ == '__main__':
    if sys.argv[1] == 'coco':
        convert_to_json(fddb_file,format='coco')
    elif sys.argv[1] == 'eval':
        convert_to_json(fddb_file,format='eval')
    elif sys.argv[1] == 'save_vid':
        if len(sys.argv) == 4:
            vid_range = (int(sys.argv[2].strip()),int(sys.argv[3].strip()))
            save_vid(save_det_dir,vid_range=vid_range)


