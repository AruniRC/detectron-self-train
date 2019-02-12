# Finds the number of images common to 2 specified JSONs

import json
import numpy as np

json1 = 'data/bdd_jsons/bdd_peds_clear_any_daytime_HP.json'
json2 = 'data/bdd_jsons/bdd_peds_clear_any_daytime_det_conf080.json'

save_file = 'common_files.txt'

with open(json1,'r') as f:
    dset1 = json.load(f)
f.close()

with open(json2,'r') as f:
    dset2 = json.load(f)
f.close()

common = []
fname1 = [img['file_name'] for img in dset1['images']]
fname2 = [img['file_name'] for img in dset2['images']]

common = list(set(fname1) - (set(fname1) - set(fname2)))

print('Number of common:',len(common))

with open(save_file,'w') as f:
    f.write('\n'.join(common))
f.close()
