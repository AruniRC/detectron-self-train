import os
import sys
import json
import os.path as osp
import numpy as np

# ------------------------------------------------------------------------------
def remove_tracklet_annots(json_file, out_file, data_set):
# ------------------------------------------------------------------------------
    ''' Remove HP JSON annotations that come from tracklets '''
    #if not osp.exists(output_dir):
    #    os.makedirs(output_dir, exist_ok=True)
    
    with open(json_file) as f:
        ann_dict = json.load(f)
    print(ann_dict.keys())

    det_annots = [x for x in ann_dict['annotations'] if x['source'] == 1]
    for ann in det_annots:
        ann['dataset'] = data_set
    ann_dict['annotations'] = det_annots
    #out_file = osp.join(output_dir, 
    #            osp.splitext(osp.basename(json_file))[0]) \
    #            + '-det' + '.json'

    print('Output: ' + out_file)
    with open(out_file, 'w', encoding='utf8') as outfile:
        outfile.write(json.dumps(ann_dict, indent=2))


if __name__ == '__main__':
    data_set = 'bdd_dets18k' #'data/bdd_jsons/bdd_dets18k.json'
    out_file = 'bdd_dets18k.json'
    json_file = 'data/bdd_jsons/bdd_HP18k.json'

    remove_tracklet_annots(json_file,out_file,data_set)


