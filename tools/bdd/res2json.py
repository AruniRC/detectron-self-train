import os
import sys
import pickle
import json
import argparse
import numpy as np
from collections import OrderedDict


def parse_args():
    parser = argparse.ArgumentParser(description='Convert results dump to training JSON')
    parser.add_argument(
                '--results_dir', required=True,
                help='Folder with results of eval')
    parser.add_argument(
                '--eval_json_file', required=True,
                help='JSON on which eval was run to generate results_dir')
    parser.add_argument(
                '--output_dir', required=True,
                help='Folder to save the created JSON')
    parser.add_argument(
                '--dataset_name', required=True,
                help='Dataset name')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    results_dir = args.results_dir # dump of eval
    output_dir = args.output_dir # folder to save created json
    dataset_name = args.dataset_name

    pred_results_dump = [f for f in os.listdir(results_dir) if f.endswith('_results.json')][0]
    json_name = pred_results_dump
    with open(os.path.join(results_dir,pred_results_dump),'r') as f:
        bbox_results = json.load(f)
    print(len(bbox_results))

    with open(args.eval_json_file,'r') as f:
        eval_json = json.load(f)

    res_json = {}
    images = eval_json['images']
    annotations = []
    categories = [{"id": 1, "name": 'pedestrian'}]
    image_count = []
    for ann_id,bbox in enumerate(bbox_results):
        image_id = bbox['image_id']
        image_count.append(image_id)
        cat_id = bbox['category_id']
        ann = {}
        ann['category_id'] = cat_id
        ann['id'] = ann_id
        ann['image_id'] = image_id
        ann['segmentation'] = []
        ann['iscrowd'] = 0
        ann['dataset'] = dataset_name
        ann['bbox'] = bbox['bbox']
        ann['area'] = ann['bbox'][2]*ann['bbox'][3]
        ann['score'] = 1
        ann['raw_score'] = bbox['score']
        annotations.append(ann)
    
    res_json['images'] = images
    res_json['categories'] = categories
    res_json['annotations'] = annotations
    
    res_json = OrderedDict(res_json)
    #print('Detected number of images:',len(set(image_count)))
    print("Num categories: %s" % len(categories))
    print("Num images: %s" % len(images))
    print("Num annotations: %s" % len(annotations))
    
    with open(os.path.join(args.output_dir,json_name), 'w', encoding='utf8') as outfile:
        outfile.write(json.dumps(res_json))
    


