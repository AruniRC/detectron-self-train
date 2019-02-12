import os
import json
import pickle
import collections

fl = 'data/bdd_jsons/bdd_dets18k.json'
#fl = 'data/bdd_jsons/bdd_HP18k.json'
#fl = 'data/bdd_jsons/bdd_HP18k_track_only.json'

#fl = 'data/bdd_jsons/bdd_peds_train.json'
#fl = 'data/bdd_jsons/bdd_peds_val.json'
#fl = 'data/bdd_jsons/bdd_peds_not_clear_any_daytime_20k_train.json'
#fl = 'data/bdd_jsons/bdd_peds_not_clear_any_daytime_val.json'
#fl = 'data/bdd_jsons/bdd_peds_dets.json'
#fl = 'data/bdd_jsons/bdd_peds_not_clear_any_daytime_train.json'
#fl = 'data/bdd_jsons/bdd_peds_clear_any_daytime_det_conf080.json'
#fl = 'data/bdd_jsons/bdd_HP18k_any_any_night.json'
#fl = 'data/bdd_jsons/bdd_peds_test.json'

#fl = 'data/cs6_jsons/train-WIDER_val-easy_conf-0.1_cs6_annot_eval_scores.json'
#fl = 'data/WIDER/wider_face_val_annot_coco_style.json'
#fl = 'data/cs6_jsons/cs6_gt_annot_val-easy.json'
#fl = 'data/cs6_jsons/cs6-test-gt.json'

#fl = 'data/bdd_jsons/vary_pure_target/bdd_peds_not_clear_any_daytime_train.json'

#fl = 'data/cityscapes_jsons/cityscapes_car_val.json'
#fl = 'data/KITTI_jsons/kitti_car_train_annot_coco_style.json'

with open(fl,'r') as f:
    j  = json.load(f)
f.close()

print('JSON:',os.path.split(fl)[-1])
print('Number of images:',len(j['images']))
print('Number of used images:',len(set([i['image_id'] for i in j['annotations']])))
print('Number of annotations:',len(j['annotations']))
print('Number of videos:',len(set([img['file_name'].split('_')[0] for img in j['images']])))


"""
# Save img_id2file_path dict
#imdir = '/mnt/nfs/work1/elm/arunirc/Data/BDD100k/images/100k/val/'
imdir = '/mnt/nfs/work1/elm/arunirc/Data/CS6_annots/'
#imdir = '/mnt/nfs/scratch1/arunirc/data/WIDER/'
img_id2file_path = {
                    img['id'] : img['file_name'] #os.path.split(img['file_name'])[-1]
                        for img in j['images']
                   }
with open(os.path.split(fl)[-1].split('.')[0]+'_ImgId2FilePath.json','w') as f:
    json.dump([imdir,collections.OrderedDict(img_id2file_path)],f)
f.close()
"""


"""
# Save list of used images -- images with annots
used_img_id_list = set([i['image_id'] for i in j['annotations']])
img_id2fname = {img['id'] : img['file_name'] for img in j['images']}
with open('imgs_with_annots_in_'+os.path.split(fl)[-1].split('.')[0]+'.txt','w') as f:
    f.write('\n'.join([img_id2fname[i] for i in used_img_id_list]))
f.close()
"""


"""
print(j['images'][:10])

k = 5000
j['images'] = j['images'][:k]
j['annotations'] = [ann for ann in j['annotations'][:10000] if ann['image_id'] < k]

#print(j['images'])
#print(j['annotations'])

with open('test_first5k.json','w') as f:
    json.dump(j,f)

f.close()
"""
