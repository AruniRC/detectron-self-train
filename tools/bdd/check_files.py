# Checks if the image paths in a JSON exist. If some do not exist, it makes another JSON, with only the images and annotations which exist
# Useful to run after the video mining FDDB-to-JSON conversion to check that the video frames have been saved properly

import os
import json
import numpy as np

"""
imdir = '/mnt/nfs/scratch1/pchakrabarty/bdd_detections_20k_fast'
#source_json = 'data/bdd_jsons/bdd_peds_clear_any_daytime_det_conf080.json'
source_json = '/mnt/nfs/scratch1/pchakrabarty/json_and_metadata_backup/fast_conf080.json'
save_json = '/mnt/nfs/scratch1/pchakrabarty/json_and_metadata_backup/fast_conf080_cleaned.json'
"""
"""
# HD Json -- initial, smaller set
imdir = '/mnt/nfs/scratch1/pchakrabarty/bdd_HP/'
source_json = '/mnt/nfs/scratch1/pchakrabarty/bdd_HP/bdd_HP.json'
save_json = '/mnt/nfs/scratch1/pchakrabarty/bdd_HP/bdd_HP_cleaned.json'
############
"""
"""
# HP 18k Json
imdir = '/mnt/nfs/scratch1/pchakrabarty/bdd_HP18k/'
source_json = '/mnt/nfs/scratch1/pchakrabarty/bdd_HP18k/bdd_HP18k.json'
save_json = '/mnt/nfs/scratch1/pchakrabarty/bdd_HP18k/bdd_HP18k_cleaned.json'
#############
"""

# HP 18k Json with different conf threshold
imdir = '/mnt/nfs/scratch1/pchakrabarty/bdd_HP18k/'
source_json = './bdd_HP18k_thresh-050.json'
save_json = './bdd_HP18k_thresh-050_cleaned.json'
#############

def img_anns(img_id,ann):
    l = [(i,a) for i,a in enumerate(ann['annotations']) if a['image_id'] == img_id]
    return l

if __name__ == '__main__':
    # load
    with open(source_json,'r') as f:
        ann = json.load(f)
    f.close()
    
    print('Building img2ann catalog...')
    #img_anns = {img['id']:[(i,a) for i,a in enumerate(ann['annotations']) if a['image_id'] == img['id']] for img in ann['images']} 
    img_anns = {img['id']:[] for img in ann['images']}
    for i,a in enumerate(ann['annotations']):
        img_anns[a['image_id']].append((i,a))
    print('...done')

    # edit path
    t = 0
    a = 0
    m = 0
    bad_img_list = []
    bad_ann_list = []
    
    for i,img in enumerate(ann['images']):
        path = img['file_name']
        print('Path:',path)
        t += 1
        #input(img)
        if not os.path.exists(os.path.join(imdir,path)):
            print('-->',os.path.exists(os.path.join(imdir,path)))
            bad_img_list.append((i,img))
            #anns = img_anns(img['id'],ann)
            anns = img_anns[img['id']]
            bad_ann_list.extend(anns)
            #print('>>>',path)
            #input(anns)
            m += 1
        else:
            a += 1
    print('Missing:',m)
    print('Available:',a)
    print('Total:',t)

    c = input('Save cleaned json to: '+str(save_json)+'?')
    if c.strip() in ['y','Y']:
        for i,b in bad_img_list[::-1]:
            del(ann['images'][i])
        for i,b in bad_ann_list[::-1]:
            del(ann['annotations'][i])
        input('Number of clean images: '+str(len(ann['images']))+'. Continue to save?')
        with open(save_json,'w') as f:
           json.dump(ann,f)
        f.close()
    


