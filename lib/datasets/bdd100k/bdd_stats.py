"""
Print out bdd100k category stats
"""

from __future__ import print_function
import os
import sys
import json
import cv2
from matplotlib import pyplot as plt

def show_stats(basedir):
    
    img_dir = os.path.join(basedir,'images','100k')
    ann_dir = os.path.join(basedir,'labels','100k')
    
    #sel_labels = ['person']
    sel_labels = []

    # select attribute values. set to [] to not restrict an attribute
    sel_attrib = {
        'weather'  :['clear'],        #clear, partly cloudy, overcast, rainy, snowy, foggy
        'scene'    :[],  #residential, highway, city street, parking lot, gas stations, tunnel
        'timeofday':['daytime']       #dawn/dusk, daytime, night
    }

    cat_inv = False
    attrib_inv = True # look at stats of settings other than selected

    for subdir in ['train','val']:
        img_subdir = os.path.join(img_dir,subdir)
        subdir = os.path.join(ann_dir,subdir)
        
        img_count = 0
        ann_count = 0
        
        lab_ann_count = {}
        lab_img_count = {}
        weather_img_count = {}
        scene_img_count = {}
        timeofday_img_count = {}
        all_lab = []

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
            frame = frames[0]
            objects = frame['objects']
            timestamp = frame['timestamp']
            
            # use only images which have the selected categories
            allowed = False
            for lab in set(t['category'] for t in objects):
                if lab in sel_labels:
                    allowed = True
                    break
            if len(sel_labels) > 0:
                if not allowed:
                    continue
            ###################
            
            # check allowed conditions
            allowed = True
            for attr in ['weather','scene','timeofday']:
                if len(sel_attrib[attr]) > 0:
                    if not (attrib[attr] in sel_attrib[attr]):
                        allowed = False
                        break
            if attrib_inv:
                allowed = (not allowed)
            if not allowed:
                continue
            ##################

            img_count += 1

            # category image count
            for lab in set([t['category'] for t in objects]):
                if lab in lab_img_count.keys():
                    lab_img_count[lab] += 1
                else:
                    lab_img_count.update({lab : 1})
                if not (lab in all_lab):
                    all_lab.append(lab)
            # category annotation count
            for i,obj in enumerate(objects):
                lab = obj['category']
                if lab in lab_ann_count.keys():
                    lab_ann_count[lab] += 1
                else:
                    lab_ann_count.update({lab : 1})
                ann_count += 1
            # weather image count
            weather = attrib['weather']
            if weather in weather_img_count.keys():
                weather_img_count[weather] += 1
            else:
                weather_img_count.update({weather : 1})
            # scene image count
            scene = attrib['scene']
            if scene in scene_img_count.keys():
                scene_img_count[scene] += 1
            else:
                scene_img_count.update({scene : 1})
            # timeofday image count
            timeofday = attrib['timeofday']
            if timeofday in timeofday_img_count.keys():
                timeofday_img_count[timeofday] += 1
            else:
                timeofday_img_count.update({timeofday : 1})

        print('Summary Stats:')
        print(os.path.split(subdir)[-1])
        print('Number of images:',img_count)
        print('Total number of annotations:',ann_count)
        
        stats = {'img_count':img_count,
                 'ann_count':ann_count,
                 'all_lab':all_lab
                }

        # category image count
        subdir_type = os.path.split(subdir)[-1]
        x = range(len(all_lab))
        y = []
        for lab in all_lab:
            y.append(lab_img_count[lab])
        ax = plt.gca()
        plt.title('Number of images per category: '+subdir_type)
        plt.bar(x,y)
        plt.xticks(x)
        ax.set_xticklabels(all_lab,rotation=90)
        plt.show()
        stats.update({'imgs_per_cat' : dict(zip(all_lab,y))})
        
        # category annotation count
        y = []
        for lab in all_lab:
            y.append(lab_ann_count[lab])
        ax = plt.gca()
        plt.title('Number of annotations per category: '+subdir_type)
        plt.bar(x,y)
        plt.xticks(x)
        ax.set_xticklabels(all_lab,rotation=90)
        plt.show()
        stats.update({'anns_per_cat' : dict(zip(all_lab,y))})

        # weather image count
        y = []
        weather_types = weather_img_count.keys()
        for weather in weather_types:
            y.append(weather_img_count[weather])
        x = range(len(y))
        ax = plt.gca()
        plt.title('Number of images over weather: '+subdir_type)
        plt.bar(x,y)
        plt.xticks(x)
        ax.set_xticklabels(weather_types,rotation=90)
        plt.show()
        stats.update({'weather_img_count':dict(zip(weather_types,y))})

        # scene image count
        y = []
        scene_types = scene_img_count.keys()
        for scene in scene_types:
            y.append(scene_img_count[scene])
        x = range(len(y))
        ax = plt.gca()
        plt.title('Number of images over scenes: '+subdir_type)
        plt.bar(x,y)
        plt.xticks(x)
        ax.set_xticklabels(scene_types,rotation=90)
        plt.show()
        stats.update({'scene_img_count':dict(zip(scene_types,y))})

        # hours image count
        y = []
        timeofday_types = timeofday_img_count.keys()
        for timeofday in timeofday_types:
            y.append(timeofday_img_count[timeofday])
        x = range(len(y))
        ax = plt.gca()
        plt.title('Number of images over time of day: '+subdir_type)
        plt.bar(x,y)
        plt.xticks(x)
        ax.set_xticklabels(timeofday_types,rotation=90)
        plt.show()
        stats.update({'timeofday_img_count':dict(zip(timeofday_types,y))})

        with open('bdd100k_'+os.path.split(subdir)[-1]+'_stats.txt','w') as f:
            f.write(str(stats))

if __name__ == '__main__':
    path = sys.argv[1]
    show_stats(path)
