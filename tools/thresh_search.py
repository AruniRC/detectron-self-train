import os
import sys
import pickle
import json
import numpy as np

import _init_paths
#from result_utils import *
from matplotlib import pyplot as plt

def thresh_search(res,class_names):
    iou_thresh = 0.5
    score_val = np.arange(0.0,1.1,0.1)
    ious = res.ious
    cat_match_thresh = {c : [] for _,c in ious.keys()}
    cat_wrong_thresh = {c : [] for _,c in ious.keys()}
    score_vs_match = { c : {t : 0 for t in score_val} for _,c in ious.keys()}
    score_vs_wrong = { c : {t : 0 for t in score_val} for _,c in ious.keys()}
    overall_match = {t : 0 for t in score_val}
    overall_wrong = {t : 0 for t in score_val}
    cat_count = {c : 0 for _,c in ious.keys()}
    overall_ann_count = 0
    for imgId,catId in ious.keys():
        iou = np.array(ious[(imgId,catId)])
        dt = res._dts[imgId,catId]
        gt = res._gts[imgId,catId]
        ndt = len(dt)
        ngt = len(gt)
        cat_count[catId] += ngt
        overall_ann_count += ngt
        for d,det in enumerate(dt):
            for g,gnd in enumerate(gt):
                dg_iou = iou[d,g]
                score = det['score']
                if dg_iou >= iou_thresh:
                    cat_match_thresh[catId].append((dg_iou,score))
                    for t in score_vs_match[catId].keys():
                        if np.abs(t-score) < 0.1:
                            score_vs_match[catId][t] += 1
                            overall_match[t] += 1
                            break
                elif dg_iou < iou_thresh:
                    cat_wrong_thresh[catId].append((dg_iou,score))
                    for t in score_vs_wrong[catId].keys():
                        if np.abs(t-score) < 0.1:
                            score_vs_wrong[catId][t] += 1
                            overall_wrong[t] += 1
                            break
    print('Total number of annotations:',overall_ann_count)
    print('Class-wise annotation count:')
    for c in cat_count.keys():
        print(c,'-->',cat_count[c])
    x = score_val
    y1 = np.array([overall_match[t] for t in score_val])
    y2 = np.array([overall_wrong[t] for t in score_val])
    y1 = y1/y1.sum()
    y2 = y2/y2.sum()
    plt.bar(x,y1,label='TP',width=0.05,alpha=0.5)
    plt.bar(x,y2,label='FP',width=0.05,alpha=0.5)
    plt.title('Overall')
    plt.legend()
    plt.show()

    for cat,var in score_vs_match.items():
        x = score_val
        y1 = np.array([score_vs_match[cat][t] for t in score_val])
        y2 = np.array([score_vs_wrong[cat][t] for t in score_val])
        # normalize to dist
        y1 = y1/y1.sum()
        y2 = y2/y2.sum()
        plt.bar(x,y1,label='TP',width=0.05,alpha=0.5)
        plt.bar(x,y2,label='FP',width=0.05,alpha=0.5)
        plt.title('Class '+str(cat)+': '+class_names[cat])
        plt.legend()
        plt.show()

if __name__ == '__main__':    
    output_dir = sys.argv[1]
    detections_pkl = os.path.join(output_dir,'detections.pkl')
    detection_results_pkl = os.path.join(output_dir,'detection_results.pkl')
    with open(detections_pkl,'rb') as f:
        det = pickle.load(f)
    with open(detection_results_pkl,'rb') as f:
        res = pickle.load(f)
    # Load class-wise splits
    class_split_dumps = [os.path.join(output_dir,fname) for fname in os.listdir(output_dir) if fname.startswith('classmAP')]
    for split_dump in class_split_dumps:
        with open(split_dump,'r') as f:
            class_map = json.load(f)
    
    class_names = [c for c in class_map.keys() if not c.startswith('IoU')]
    thresh_search(res,class_names)


