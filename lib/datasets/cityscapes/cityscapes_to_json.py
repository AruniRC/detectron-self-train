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
    x,y = zip(*poly)
    x1,x2 = min(x),max(x)
    y1,y2 = min(y),max(y)
    return ((x1,y1),(x2,y2))

def cat2id(lab):
    cat2id_map = {'car':1,'person':2,'truck':3,'bus':4,'motorcycle':5,'bicycle':6,'rider':7} # final 7 classes
    
    #cat2id_map = {'car':1,'traffic light':2,'person':3,'motorcycle':4,'bus':5} # initial 5 classes
    
    return cat2id_map[lab]

def genJSON(basedir):
    sel_labels = ['car','person','truck','bus','motorcycle','bicycle','rider'] # final 7 classes
    #sel_labels = ['person']
    #sel_labels = ['car','traffic light','person','motorcycle','bus'] # initial 5 classes
    
    img_dir = os.path.join(basedir,'leftImg8bit')
    ann_dir = os.path.join(basedir,'gtFine')
    #flat_dir = os.path.join(basedir,'all_imgs')
    
    #categories = [{'id':(i+1),'name':cat} for i,cat in enumerate(sel_labels)]
    categories = [{'id':cat2id(cat),'name':cat} for cat in sel_labels]
    
    for subdir in ['train','val']:#os.listdir(ann_dir):
        img_subdir = os.path.join(img_dir,subdir)
        subdir = os.path.join(ann_dir,subdir)
        
        images = []
        annotations = []
        ann_dict = {}
        img_id = 0
        ann_id = 0
        
        for city in os.listdir(subdir):
            
            img_city = os.path.join(img_subdir,city)
            city = os.path.join(subdir,city)
            
            img_samples = os.listdir(img_city)
            samples = os.listdir(city)
            
            img_files   = sorted([os.path.join(img_city,s) for s in img_samples if s.split('.')[0].endswith('leftImg8bit')])
            poly_files  = sorted([os.path.join(city,s) for s in samples if s.split('.')[0].endswith('polygons')])
            col_files   = sorted([os.path.join(city,s) for s in samples if s.split('.')[0].endswith('color')])
            inst_files  = sorted([os.path.join(city,s) for s in samples if s.split('.')[0].endswith('instanceIds')])
            lbid_files  = sorted([os.path.join(city,s) for s in samples if s.split('.')[0].endswith('labelIds')])
            
            for filename,poly,col,_,_, in zip(img_files,poly_files,col_files,inst_files,lbid_files):
                with open(poly,'r') as f:
                    poly_data = json.load(f)
                bbox_data = poly_data.copy()
                ignore_obj = []
                image = {}
                image['width'] = poly_data['imgWidth']
                image['height'] = poly_data['imgHeight']
                image['id'] = img_id
                img_id += 1
                
                #img = cv2.imread(filename)

                filename = os.path.join(img_dir,os.path.split(subdir)[-1],os.path.split(city)[-1],os.path.split(filename)[-1]) # use the full path
                
                #filename = os.path.split(filename)[-1] # using just the filename
                
                print('Reading:',filename)
                image['file_name'] = filename

                has_obj = False
                for i,obj in enumerate(poly_data['objects']):
                    lab = obj['label']
                    if not (lab in sel_labels):
                        ignore_obj.append(i)
                        continue
                    has_obj = True
                    ann = {}
                    ann['id'] = ann_id
                    ann_id += 1
                    ann['image_id'] = image['id']
                    ann['segmentation'] = []
                    ann['category_id'] = cat2id(lab)
                    ann['iscrowd'] = 0
                    bbox = poly2bbox(obj['polygon'])
                    ann['bbox'] = [bbox[0][0],bbox[0][1],bbox[1][0]-bbox[0][0],bbox[1][1]-bbox[0][1]]
                    ann['area'] = (bbox[1][0]-bbox[0][0])*(bbox[1][1]-bbox[0][1])
                    annotations.append(ann)
                
                    #cv2.rectangle(img,bbox[0],bbox[1],(0,255,0),3)
                
                images.append(image)
                #if has_obj:
                #    images.append(image)
                
                '''if not has_obj:
                    print('>>>',[l['label'] for l in poly_data['objects']])
                    img = cv2.imread(filename)
                    img = cv2.resize(img,(800,800))
                    plt.imshow(img)
                    plt.show()
                '''
                for i in ignore_obj[::-1]:
                    del(bbox_data['objects'][i])
                #print('>>>',bbox_data)
                #raw_input('??')
        ann_dict['images'] = images
        ann_dict['categories'] = categories
        ann_dict['annotations'] = annotations
        print(subdir,len(annotations))
        #print('>>>',annotations)
        with open('cityscapes_'+os.path.split(subdir)[-1].strip()+'.json','w',encoding='utf8') as f:
            f.write(json.dumps(ann_dict))

if __name__ == '__main__':
    path = sys.argv[1]
    save_path = sys.argv[2] if (len(sys.argv) > 2) else ''
    genJSON(path)
