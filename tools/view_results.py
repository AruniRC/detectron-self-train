# Read the results dump from tools/test_net.py and print the results

import os
import sys
import pickle
import json
import numpy as np

def iou_thresh(res,iou):
    # Check IoU computation: give image ID, category ID
    images = res.params.imgIds
    categories = res.params.catIds
    for cat in categories:
        for img in images:
            iou_mat = res.computeIoU(0,cat)
            # compute true positives
            # compute false positives
    
    #print(iou_mat)

if __name__ == '__main__':
    
    output_dir = sys.argv[1]
    detection_results_pkl = os.path.join(output_dir,'detection_results.pkl')
    with open(detection_results_pkl,'rb') as f:
        res = pickle.load(f)
    
    # Load class-wise splits
    class_split_dumps = [os.path.join(output_dir,fname) for fname in os.listdir(output_dir) if fname.startswith('classmAP')]
    for split_dump in class_split_dumps:
        with open(split_dump,'r') as f:
            class_map = json.load(f)
        f.close()
        print('\n\tClass-wise Split with IoU@'+str(class_map['IoU_low'])+':'+str(class_map['IoU_high']))
        for cls_ind,cls in enumerate(class_map.keys()):
            if cls.startswith('IoU'):
                continue
            print('\t',cls,'\t|',class_map[cls])
            
            #if not (cls == 'Overall'):
            #    print('>>>',dir(res),len(res.ious),res.eval['recall'].shape)
            #    input(np.mean(res.eval['recall'][:, cls_ind - 3, 0, 2]))
    
    # Summary metrics
    print('\n\t\t~~~~~ Summary Metrics ~~~~~~')
    res.summarize()

    '''
    with open('./test_res/bbox_cityscapes_val_results.json','r') as f:
        bbox_results = json.load(f)
    print(len(bbox_results))

    for bbox in bbox_results:
        image_id = bbox['image_id']
        cat_id = bbox['category_id']
        bbox_id = bbox['bbox']
        score = bbox['score']
        print('>>>',score)
    '''

