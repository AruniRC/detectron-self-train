"""
Print out bdd100k category stats
"""

from __future__ import print_function
import os
import sys
import json
import cv2
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

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
    attrib_inv = False # look at stats of settings other than selected
    
    ext = '.pdf'
    save_stats_dict = {}

    for subdir in ['train']:#,'val']:
        img_subdir = os.path.join(img_dir,subdir)
        subdir = os.path.join(ann_dir,subdir)
        
        img_count = 0
        ann_count = 0
        
        lab_ann_count = {}
        lab_img_count = {}
        weather_img_count = {'clear':0,'rainy':0,'snowy':0,'overcast':0,'partly cloudy':0,'foggy':0,'undefined':0}
        scene_img_count = {'highway':0,'city street':0,'parking lot':0,'tunnel':0,'residential':0,'gas stations':0,'undefined':0}
        timeofday_img_count = {'daytime':0,'night':0,'dawn/dusk':0,'undefined':0}
        all_lab = []


        img_samples = os.listdir(img_subdir)
        samples = os.listdir(subdir)
        

        """####### tmp: Stats for the BDD HP18k Json #######
        with open('../../../data/bdd_jsons/bdd_HP18k.json','r') as f:
            d = json.load(f)
        f.close()
        img_samples = [i['file_name'].split('_')[0]+'.jpg' for i in d['images']]
        samples = [i.split('.')[0]+'.json' for i in img_samples]
        ###################################
        """


        img_files  = sorted([os.path.join(img_subdir,s) for s in img_samples])
        lab_files  = sorted([os.path.join(subdir,s) for s in samples])
        
        for img_file,lab_file in list(zip(img_files,lab_files)):
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
        subdir_type = os.path.split(subdir)[-1]

        # category image count
        f = plt.figure()
        x = range(len(all_lab))
        y = []
        for lab in all_lab:
            y.append(lab_img_count[lab])
        ax = plt.gca()
        plt.title('Number of images per category: '+subdir_type)
        plt.bar(x,y)
        plt.xticks(x)
        ax.set_xticklabels(['\n'.join(t.strip().split(' ')) for t in all_lab],fontsize=9)#,rotation=90)
        #plt.show()
        f.savefig('category_img_count_'+str(subdir_type)+ext,bbox_inches='tight')
        stats.update({'imgs_per_cat' : dict(zip(all_lab,y))})
        
        # category annotation count
        f = plt.figure()
        y = []
        for lab in all_lab:
            y.append(lab_ann_count[lab])
        ax = plt.gca()
        plt.title('Number of annotations per category: '+subdir_type)
        plt.bar(x,y)
        plt.xticks(x)
        ax.set_xticklabels(['\n'.join(t.strip().split(' ')) for t in all_lab],fontsize=9)#,rotation=90)
        #plt.show()
        f.savefig('category_ann_count_'+str(subdir_type)+ext,bbox_inches='tight')
        stats.update({'anns_per_cat' : dict(zip(all_lab,y))})
        
        # weather image count
        y = []
        weather_types = weather_img_count.keys()
        for weather in weather_types:
            y.append(weather_img_count[weather])
        x = range(len(y))
        f = plt.figure()
        ax = plt.gca()
        #plt.title('Number of images over weather: '+subdir_type)
        plt.title('Weather',fontsize=18)
        plt.ylabel('number of images',fontsize=16)
        ax.yaxis.set_tick_params(labelsize=16)
        plt.bar(x,y,width=0.8)
        plt.xticks(x)
        ax.set_xticklabels(['\n'.join(t.strip().split(' ')) for t in weather_types],fontsize=16,rotation=45)
        f.savefig('weather_img_count_'+str(subdir_type)+ext,bbox_inches='tight')
        stats.update({'weather_img_count':dict(zip(weather_types,y))})

        # scene image count
        y = []
        scene_types = scene_img_count.keys()
        for scene in scene_types:
            y.append(scene_img_count[scene])
        x = range(len(y))
        f = plt.figure()
        ax = plt.gca()
        #plt.title('Number of images over scenes: '+subdir_type)
        plt.title('Scenes',fontsize=18)
        ax.yaxis.set_tick_params(labelsize=16)
        plt.bar(x,y)
        plt.xticks(x)
        ax.set_xticklabels(['\n'.join(t.strip().split(' ')) for t in scene_types],fontsize=9)#scene_types)#,rotation=90)
        #plt.show()
        f.savefig('scene_img_count_'+str(subdir_type)+ext,bbox_inches='tight')
        stats.update({'scene_img_count':dict(zip(scene_types,y))})

        # hours image count
        y = []
        timeofday_types = timeofday_img_count.keys()
        for timeofday in timeofday_types:
            y.append(timeofday_img_count[timeofday])
        x = range(len(y))
        f = plt.figure()
        ax = plt.gca()
        #plt.title('Number of images over time of day: '+subdir_type)
        plt.title('Time of Day',fontsize=18)
        ax.yaxis.set_tick_params(labelsize=16)
        plt.bar(x,y,width=0.4)
        plt.xticks(x)
        ax.set_xticklabels(['\n'.join(t.strip().split(' ')) for t in timeofday_types],fontsize=16,rotation=45)
        f.savefig('time_img_count_'+str(subdir_type)+ext,bbox_inches='tight')
        stats.update({'timeofday_img_count':dict(zip(timeofday_types,y))})
        
        if save_stats_dict:
            with open('bdd100k_'+os.path.split(subdir)[-1]+'_stats.txt','w') as f:
                f.write(str(stats))

if __name__ == '__main__':
    path = sys.argv[1]
    show_stats(path)
