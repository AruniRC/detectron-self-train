import os
import sys
import pickle
import json
import numpy as np
import argparse

import _init_paths
#from result_utils import *
from matplotlib import pyplot as plt

def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--output_dir', required=True,
        help='Output dir containing detections.pkl')
    parser.add_argument(
        '--score_map',
        default='',
        help='NumPy file containing a map to change the score distribution')
    parser.add_argument(
        '--map_TP_only',
        action='store_true',
        help='apply score mapping only to the True Positive histogram')
    parser.add_argument(
        '--map_FP_only',
        action='store_true',
        help='apply score mapping only to the False Positive histogram')
    parser.add_argument(
        '--show_TP_only',
        action='store_true',
        help='show histogram of only True Positive probabilities')
    parser.add_argument(
        '--show_FP_only',
        action='store_true',
        help='show histogram of only False Positive probabilities')
    parser.add_argument(
        '--save_hist',
        default='',
        help='Path to a NumPy file to save the histograms')
    parser.add_argument(
        '--save_det',
        default='',
        help='Path to a pickle file to save the detections')

    return parser.parse_args()


def thresh_search(res,class_names,
                    score_map=None,
                    map_TP_only=False,
                    map_FP_only=False,
                    show_TP_only=False,
                    show_FP_only=False,
                    save_hist_file='',
                    save_det_file=''):
    iou_thresh = 0.5
    step = 0.01
    score_val = np.arange(0.0,1.1,step)
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
        #print('>>>',ndt,ngt,iou.shape)
        cat_count[catId] += ngt
        overall_ann_count += ngt
        for d,det in enumerate(dt):
            #print('here...',iou_thresh,det)
            score = det['score']

            d_match_count = 0
            for g,gnd in enumerate(gt):
                dg_iou = iou[d,g]
                # found a match -- true positive
                if dg_iou >= iou_thresh:
                    d_match_count += 1

                    # apply score mapping on true positives
                    if not map_FP_only:
                        if not (score_map is None):
                            s,ms = score_map
                            score = ms[np.argmin(np.abs(s-score))]
                    ############

                    cat_match_thresh[catId].append((dg_iou,score))
                    for t in score_vs_match[catId].keys():
                        if np.abs(t-score) < step:
                            score_vs_match[catId][t] += 1
                            overall_match[t] += 1
                            break
            # d did not match any gt box in the image -- false positive
            if d_match_count == 0:
                
                # apply score mapping on false positives
                if not map_TP_only:
                    if not (score_map is None):
                        s,ms = score_map
                        score = ms[np.argmin(np.abs(s-score))]
                ############
                
                for t in score_vs_wrong[catId].keys():
                    if np.abs(t-score) < step:
                        score_vs_wrong[catId][t] += 1
                        overall_wrong[t] += 1
                        break
    print('Total number of annotations:',overall_ann_count)
    print('Class-wise annotation count:')
    for c in cat_count.keys():
        print(c,'-->',cat_count[c])
    
    
    # Display histograms
    x = score_val
    y1 = np.array([overall_match[t] for t in score_val])
    y2 = np.array([overall_wrong[t] for t in score_val])
    y1 = y1/y1.sum()
    y2 = y2/y2.sum()
    
    plt.ylim((0,0.4))
    
    lab = os.path.split(output_dir)[-1]
    if len(lab) == 0:
        lab = os.path.split(output_dir)[-2]
    
    """f = plt.figure()
    plt.ylim((0,0.12))
    name = 'Source domain.pdf'
    plt.title(name.split('.')[0],fontsize=16)
    vline = np.arange(0,0.12,0.05)
    plt.plot([0.5]*len(vline),vline,'k--')
    plt.ylabel('normalized frequency',fontsize=16)
    """
    if not show_FP_only:
        plt.bar(x,y1,label=lab+'_TP',width=step,alpha=0.5)
        #plt.bar(x,y1,label='TP',width=step,alpha=0.5)
    if not show_TP_only:
        plt.bar(x,y2,label=lab+'_FP',width=step,alpha=0.5)
        #plt.bar(x,y2,label='FP',width=step,alpha=0.5)
    plt.xlabel('scores',fontsize=0.5)
    plt.legend(fontsize=16)
    plt.show()
    """
    f.savefig(os.path.join('noise_red_soft_labels_plots',name))
    """
    # Display CDF
    y1_cdf = np.cumsum(y1)
    y2_cdf = np.cumsum(y2)
    
    plt.ylim((0,1))
    plt.title('Y1')
    if not show_FP_only:
        plt.plot(x,y1_cdf,label='TP_CDF')
    if not show_TP_only:
        plt.plot(x,y2_cdf,label='FP_CDF')
    plt.xlabel('scores')
    plt.title('CDF')
    plt.legend()
    plt.show()
    
    # Chi-squared distance between
    tp_fp_chi_squared = np.sum( ((y1-y2)**2)/(y1+y2+1e-6) )
    print('Chi squared distance between TP and FP:',tp_fp_chi_squared)

    
    # Save the raw TP and FP histograms
    if len(save_hist_file) > 0:
        np.save(save_hist_file,[x,y1,y2]) # X, TP, FP

    # Save detection outputs
    if len(save_det_file) > 0:
        iou_thresh = 0.5
        def add_tp_label(d,det,gt,iou):
            det.update({'TP' : False})
            for g,gnd in enumerate(gt):
                dg_iou = iou[d,g]
                if dg_iou >= iou_thresh:
                    det['TP'] = True
            #print('-->',det)
            return det
        all_dets = {(imgId,catId) : 
                        [add_tp_label(d,det,res._gts[imgId,catId],ious[(imgId,catId)]) \
                            for d,det in enumerate(res._dts[imgId,catId])] \
                                for imgId,catId in ious.keys()}
        #print('>>>',all_dets)
        with open(save_det_file,'wb') as f:
            pickle.dump(all_dets,f)
        f.close()
    """
    # Class-specific TP, FP histograms
    for cat,var in score_vs_match.items():
        x = score_val
        y1 = np.array([score_vs_match[cat][t] for t in score_val])
        y2 = np.array([score_vs_wrong[cat][t] for t in score_val])
        # normalize to dist
        y1 = y1/y1.sum()
        y2 = y2/y2.sum()
        plt.bar(x,y1,label='TP',width=step,alpha=0.5)
        plt.bar(x,y2,label='FP',width=step,alpha=0.5)
        plt.title('Class '+str(cat)+': '+class_names[cat])
        plt.legend()
        plt.show()
    """


if __name__ == '__main__':    
    
    args = parse_args()
    output_dir = args.output_dir

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
    
    # load score map
    score_map_file = args.score_map
    if len(score_map_file) == 0:
        score_map = None
    else:
        score_map = np.load(score_map_file)
    
    thresh_search(res,class_names,
                    score_map=score_map,
                    map_TP_only=args.map_TP_only,
                    map_FP_only=args.map_FP_only,
                    show_TP_only=args.show_TP_only,
                    show_FP_only=args.show_FP_only,
                    save_hist_file=args.save_hist,
                    save_det_file=args.save_det)


