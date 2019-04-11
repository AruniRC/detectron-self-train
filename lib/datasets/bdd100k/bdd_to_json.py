"""
Save CityScape segmentation labels as bounding boxes in JSON

Usage:
        python cityscape2json.py <path to ground truth folder>
"""

from __future__ import print_function
import os
import sys
import json
import cv2
from matplotlib import pyplot as plt

def poly2bbox(poly):
    x,y,_ = zip(*poly)
    x1,x2 = min(x),max(x)
    y1,y2 = min(y),max(y)
    #print(poly)
    #print(x1,x2,y1,y2)
    #input('??')
    return ((x1,y1),(x2,y2))

def cat2id(lab):
    #cat2id_map = {'car':1,'person':2,'truck':3,'bus':4,'motorcycle':5,'bicycle':6,'rider':7} # final 7 classes
    cat2id_map = {'person':1} # only pedestrians
    #cat2id_map = {'car':1,'traffic light':2,'person':3,'motorcycle':4,'bus':5} # 5 initial classes
    #cat2id_map = {'car':1,'person':2,'truck':3,'bus':4,'motorcycle':5,'bicycle':6,'traffic light':7,'sign':8,'rider':9} # all classes
    return cat2id_map[lab]

def bdd2cityscapes(label):
    if label == 'motor':
        return 'motorcycle'
    elif label == 'bike':
        return 'bicycle'
    else:
        return label

def attrib2str(attrib_dict):
    keys = []
    for k in attrib_dict.keys():
        k_val = ','.join(attrib_dict[k])
        if len(k_val) == 0:
            k_val = 'Any'
        keys.append(k+'-'+k_val)
    return '_'.join(keys)

def genJSON(basedir):
    #################### PARAMS ###############
    # select labels
    #sel_labels = ['car','person','truck','bus','motorcycle','bicycle','rider'] # final 7 classes
    sel_labels = ['person'] # just pedestrians
    #sel_labels = ['car','traffic light','person','motorcycle','bus'] # initial 5 classes
    #sel_labels = ['car','person','truck','bus','motorcycle','bicycle','traffic light','sign','rider']

    # select attribute values. set to [] to not restrict an attribute
    sel_attrib = {
                  'weather'  :[],        #clear, partly cloudy, overcast, rainy, snowy, foggy
                  'scene'    :['city street'],  #residential, highway, city street, parking lot, gas stations, tunnel
                  'timeofday':['daytime']       #dawn/dusk, daytime, night
                 }

    # file with list of specific videos (used for detection)
    filelist = ''
    #filelist = '/mnt/nfs/scratch1/ashishsingh/FALL2018/BDD20k/bdd_target_20k.txt'
    
    # set to true if you want to select the images that do NOT satisfy the constraints set in sel_attrib
    choose_inverted_attrib = False

    ################### END of PARAMS ###########

    img_dir = os.path.join(basedir,'images','100k')
    ann_dir = os.path.join(basedir,'labels','100k')
    vid_dir = os.path.join(basedir,'videos','100k')

    categories = [{'id':cat2id(cat),'name':cat} for cat in sel_labels]
    
    for subdir in ['train','val']:
        img_file_list = []
        vid_file_list = []

        img_subdir = os.path.join(img_dir,subdir)
        subdir = os.path.join(ann_dir,subdir)
        
        images = []
        annotations = []
        ann_dict = {}
        img_id = 0
        ann_id = 0

        if len(filelist) == 0:
            img_samples = os.listdir(img_subdir)
            samples = os.listdir(subdir)
        else:
            # load filenames from a file
            with open(filelist,'r') as f:
                flist = f.readlines()
            f.close()
            # if given a list of videos, choose the corresponding annotated frame files
            samples = []
            for i in range(len(flist)):
                flist[i] = flist[i].strip()
                if flist[i].endswith('.mov'):
                    flist[i] = flist[i][:-4]+'.jpg'
                samples.append(flist[i][:-4]+'.json') # corresponding annotation json
            img_samples = flist
        
        img_files  = sorted([os.path.join(img_subdir,s) for s in img_samples])
        lab_files  = sorted([os.path.join(subdir,s) for s in samples])
        for img_file,lab_file in zip(img_files,lab_files):
            with open(lab_file,'r') as f:
                data = json.load(f)
            
            name = data['name']
            attrib = data['attributes']
            frames = data['frames']

            # check allowed conditions
            allowed = True
            for attr in ['weather','scene','timeofday']:
                if len(sel_attrib[attr]) > 0:
                    if not (attrib[attr] in sel_attrib[attr]):
                        allowed = False
                        break
            if choose_inverted_attrib:
                allowed = (not allowed) # invert allowed constraints
            if not allowed:
                continue
            
            frame = frames[0]
            timestamp = frame['timestamp']
            objects = frame['objects']
            image = {}
            image['width'] = 720
            image['height'] = 1280
            image['id'] = img_id
            img_id += 1
            print('Reading:',img_file)
            image['file_name'] = img_file
            images.append(image)
            
            img_file_list.append(img_file)
            vid_file_list.append(os.path.join(
                                        vid_dir,
                                        os.path.split(subdir)[-1],
                                        os.path.split(img_file)[-1].split('.')[0]+'.mov'))
            
            for i,obj in enumerate(objects):
                lab = obj['category']
                lab = bdd2cityscapes(lab)
                obj_attrib = obj['attributes']
                if 'box2d' in obj:
                    bbox = obj['box2d']
                elif 'poly2d' in obj:
                    poly = obj['poly2d']
                    bbox = poly2bbox(poly)
                if not (lab in sel_labels):
                    continue
                ann = {}
                ann['id'] = ann_id
                ann_id += 1
                ann['image_id'] = image['id']
                ann['segmentation'] = []
                ann['category_id'] = cat2id(lab)
                ann['iscrowd'] = 0
                ann['bbox'] = list(map(int, [bbox['x1'],bbox['y1'],bbox['x2']-bbox['x1'],bbox['y2']-bbox['y1']] ))
                bbox = ann['bbox']
                ann['area'] = bbox[2]*bbox[3]
                annotations.append(ann)

        ann_dict['images'] = images
        ann_dict['categories'] = categories
        ann_dict['annotations'] = annotations
        #print(subdir,len(annotations))
        with open('bdd_'+attrib2str(sel_attrib)+'_'+os.path.split(subdir)[-1].strip()+'.json','w',encoding='utf8') as f:
            f.write(json.dumps(ann_dict,indent=2))
        f.close()
        with open('img_files_'+attrib2str(sel_attrib)+'_'+os.path.split(subdir)[-1].strip()+'.txt','w',encoding='utf8') as f:
            f.write('\n'.join(img_file_list)+'\n')
        with open('vid_files_'+attrib2str(sel_attrib)+'_'+os.path.split(subdir)[-1].strip()+'.txt','w',encoding='utf8') as f:
            f.write('\n'.join(vid_file_list)+'\n')
        f.close()

if __name__ == '__main__':
    path = sys.argv[1]
    genJSON(path)
