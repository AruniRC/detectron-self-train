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

def show_stats(basedir):
    #sel_labels = ['car','traffic light','person','motorcycle','bus']
    img_dir = os.path.join(basedir,'leftImg8bit')
    ann_dir = os.path.join(basedir,'gtFine')
    
    for subdir in ['train','val']:
        img_subdir = os.path.join(img_dir,subdir)
        subdir = os.path.join(ann_dir,subdir)
        
        img_count = 0
        ann_count = 0
        
        lab_ann_count = {}
        lab_img_count = {}
        all_lab = []

        for city in os.listdir(subdir):
            
            img_city = os.path.join(img_subdir,city)
            city = os.path.join(subdir,city)
            
            img_samples = os.listdir(img_city)
            samples = os.listdir(city)
            
            img_files   = sorted([os.path.join(img_city,s) for s in img_samples if s.split('.')[0].endswith('leftImg8bit')])
            poly_files  = sorted([os.path.join(city,s) for s in samples if s.split('.')[0].endswith('polygons')])
            
            for filename,poly in zip(img_files,poly_files):
                with open(poly,'r') as f:
                    poly_data = json.load(f)
                bbox_data = poly_data.copy()
                img_count += 1
                
                filename = os.path.join(img_dir,os.path.split(subdir)[-1],os.path.split(city)[-1],os.path.split(filename)[-1])
                print('Reading:',filename)

                for lab in set([t['label'] for t in poly_data['objects']]):
                    if lab in lab_img_count:
                        lab_img_count[lab] += 1
                    else:
                        lab_img_count.update({lab : 1})
                    if not (lab in all_lab):
                        all_lab.append(lab)
                    
                for i,obj in enumerate(poly_data['objects']):
                    lab = obj['label']
                    if lab in lab_ann_count:
                        lab_ann_count[lab] += 1
                    else:
                        lab_ann_count.update({lab : 1})
                    ann_count += 1
        
        print('Summary Stats:')
        print(os.path.split(subdir)[-1])
        print('Number of images:',img_count)
        print('Total number of annotations:',ann_count)
        
        stats = {'img_count':img_count,
                 'ann_count':ann_count,
                 'all_lab':all_lab
                }

        #print('Number of images per label:')
        x = range(len(all_lab))
        y = []
        for lab in all_lab:
            #print('\t',lab,lab_img_count[lab])
            y.append(lab_img_count[lab])
        ax = plt.gca()
        plt.title('Number of images per category')
        plt.bar(x,y)
        plt.xticks(x)
        ax.set_xticklabels(all_lab,rotation=90)
        plt.show()

        stats.update({'imgs_per_cat' : dict(zip(all_lab,y))})
        
        #print('Number of instances per category:')
        y = []
        for lab in all_lab:
            #print('\t',lab,lab_ann_count[lab])
            y.append(lab_ann_count[lab])
        ax = plt.gca()
        plt.title('Number of annotations per category')
        plt.bar(x,y)
        plt.xticks(x)
        ax.set_xticklabels(all_lab,rotation=90)
        plt.show()

        stats.update({'anns_per_cat' : dict(zip(all_lab,y))})
        
        with open('cityscapes_gtFine_'+os.path.split(subdir)[-1]+'_stats.txt','w') as f:
            f.write(str(stats))

if __name__ == '__main__':
    path = sys.argv[1]
    show_stats(path)
