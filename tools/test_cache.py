import numpy as np
import _init_paths
import pickle

cache1_file = 'data/cache_gt-scores_joint/cs6-train-hp_gt_roidb.pkl'
cache2_file = 'data/cache_gt-scores_joint/cs6-train-hp_gt_roidb_backup.pkl'

with open(cache1_file,'rb') as f:
    cache1 = pickle.load(f)
f.close()

with open(cache2_file,'rb') as f:
    cache2 = pickle.load(f)
f.close()


for c1,c2 in zip(cache1,cache2):
    print(c1['image'],c2['image'],np.sum(c1['boxes']-c2['boxes']),(c1['id']-c2['id']))


