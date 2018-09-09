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
    cat2id_map = {'car':1,'traffic light':2,'person':3,'motorcycle':4,'bus':5}
    return cat2id_map[lab]

def genJSON(basedir):
    # select labels
    sel_labels = ['car','traffic light','person','motorcycle','bus']
    
    # select attribute values. set to [] to not restrict an attribute
    sel_attrib = {
                  'weather'  :[],        #clear, partly cloudy, overcast, rainy, snowy, foggy
                  'scene'    :[],  #residential, highway, city street, parking lot, gas stations, tunnel
                  'timeofday':['night']       #dawn/dusk, daytime, night
                 }
    img_dir = os.path.join(basedir,'images','100k')
    ann_dir = os.path.join(basedir,'labels','100k')
    
    categories = [{'id':cat2id(cat),'name':cat} for cat in sel_labels]
    
    for subdir in ['train','val']:
        img_subdir = os.path.join(img_dir,subdir)
        subdir = os.path.join(ann_dir,subdir)
        
        images = []
        annotations = []
        ann_dict = {}
        img_id = 0
        ann_id = 0

        img_samples = os.listdir(img_subdir)
        samples = os.listdir(subdir)
        
        img_files   = sorted([os.path.join(img_subdir,s) for s in img_samples])
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
            for i,obj in enumerate(objects):
                lab = obj['category']
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
                annotations.append(ann)

        ann_dict['images'] = images
        ann_dict['categories'] = categories
        ann_dict['annotations'] = annotations
        #print(subdir,len(annotations))
        with open('bdd100k_'+str(sel_attrib)+'_'+os.path.split(subdir)[-1].strip()+'.json','w',encoding='utf8') as f:
            f.write(json.dumps(ann_dict))

if __name__ == '__main__':
    path = sys.argv[1]
    genJSON(path)
