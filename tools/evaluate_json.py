import os
import sys
import cv2
import json
import pickle
import logging
import importlib
import numpy as np

from pycocotools.cocoeval import COCOeval

import _init_paths
from datasets.json_dataset import JsonDataset
import datasets.json_dataset_evaluator as json_dataset_evaluator
from datasets.json_dataset_evaluator import evaluate_boxes
from datasets import dataset_catalog
from datasets import task_evaluation

#gt_dataset_name = 'cs6_annot_eval_val-easy'
gt_dataset_name = 'cs6_train_gt'

# CS6 Ground Truth -- full dataset (NOT prediction from a model: actual ground truth for the full dataset)
cs6_gt_json = '/mnt/nfs/work1/elm/pchakrabarty/cs6_jsons/cs6-train-gt_face_train_annot_coco_style.json'

# Ground truth JSON (NOT prediction from a model: actualy ground truth)
cs6_gt_val_easy_json = '/mnt/nfs/work1/elm/pchakrabarty/cs6_jsons/cs6_gt_annot_val-easy.json'

# WIDER baseline model detections
wider_model_det_json = '/mnt/nfs/work1/elm/pchakrabarty/cs6_jsons/train-WIDER_val-easy_conf-0.1_cs6_annot_eval_scores.json'
     
# CS6 GT model detections
cs6_gt_model_det_json = '/mnt/nfs/work1/elm/pchakrabarty/cs6_jsons/train-CS6-GT-all-30k_val-easy_conf-0.1_cs6_annot_eval_scores.json'
         
# CS6 GT+WIDER model detections
cs6_hp_and_wider_bs64_model_det_json = '/mnt/nfs/work1/elm/pchakrabarty/cs6_jsons/train-CS6-HP+WIDER-bs64-5k_val-easy_conf-0.1_cs6_annot_eval_scores.json'

# CS6 train HP + WIDER bs512 model detections
cs6_hp_and_wider_bs512_model_det_json = '/mnt/nfs/work1/elm/pchakrabarty/cs6_jsons/train-CS6-HP+WIDER-bs512-15k_val-easy_conf-0.1_cs6_annot_eval_scores.json'
 
# CS6 train HP
cs6_hp_model_det_json = '/mnt/nfs/work1/elm/pchakrabarty/cs6_jsons/train-CS6-Train-HP-5k_val-easy_conf-0.1_cs6_annot_eval_scores.json'

# CS6 train HP+WIDER (bs512-gpu4-5k) model detections
cs6_hp_and_wider_bs512_gpu4_5k_model_det_json = '/mnt/nfs/work1/elm/pchakrabarty/cs6_jsons/train-CS6-HP+WIDER-bs512-gpu4-5k_val-easy_conf-0.1_cs6_annot_eval_scores.json'


det_json = cs6_gt_json

output_dir = 'tmp'

def disp_detection_eval_metrics(json_dataset, coco_eval, iou_low=0.5, iou_high=0.95, output_dir=None):
     def _get_thr_ind(coco_eval, thr):
         ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                        (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
         iou_thr = coco_eval.params.iouThrs[ind]
         assert np.isclose(iou_thr, thr)
         return ind
 
     IoU_lo_thresh = iou_low
     IoU_hi_thresh = iou_high
 
     ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
     ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
 
     class_maps = {}
     class_maps['IoU_low'] = IoU_lo_thresh
     class_maps['IoU_high'] = IoU_hi_thresh
 
     precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
     ap_default = np.mean(precision)#[precision > -1])
     recall = coco_eval.eval['recall'][ind_lo:(ind_hi + 1), :, 0, 2]
     ar_default = np.mean(recall)#[recall > -1])
     print(
         '~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] ~~~~'.format(
             IoU_lo_thresh, IoU_hi_thresh))
     print('Overall --> {:.2f},{:.2f}'.format(100 * ap_default, 100 * ar_default))
     class_maps.update({'Overall' : 100*ap_default})
     for cls_ind, cls in enumerate(json_dataset.classes):
         if cls == '__background__':
             continue
         # minus 1 because of __background__
         precision = coco_eval.eval['precision'][
             ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
         ap = np.mean(precision[precision > -1])
         recall = coco_eval.eval['recall'][ind_lo:(ind_hi+1), cls_ind - 1, 0, 2]
         ar = np.mean(recall[recall > -1])
         print(str(cls)+' --> {:.2f},{:.2f}'.format(100 * ap, 100 * ar))
         class_maps.update({str(cls) : 100 * ap})
 
     # save class-wise mAP
     if not (output_dir is None):
         with open(os.path.join(output_dir,'classmAP@IoUs'+str(iou_low)+'-'+str(iou_high)+'.json'),'w') as f:
             json.dump(class_maps,f)
         f.close()
 
     print('~~~~ Summary metrics ~~~~')
     coco_eval.summarize()


def empty_results(num_classes,num_images):
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    all_segms = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    all_keyps = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    return all_boxes, all_segms, all_keyps

def get_by_filename(det_json,fname):
    for det_f in det_json['images']:
        if det_f['file_name'] == fname:
            return (det_f['id'],det_f)

def get_boxes_by_img_id(det_json,img_id):
    boxes = []
    for det_ann in det_json['annotations']:
        if det_ann['image_id'] == img_id:
            boxes.append(det_ann)
    return boxes

def eval_json(det_json,gt_json):
    json_dataset = JsonDataset(gt_dataset_name)
    gt_json = dataset_catalog.DATASETS[gt_dataset_name]['annotation_file']
    with open(det_json,'rb') as f:
        det = json.load(f)
    f.close()
    with open(gt_json,'rb') as f:
        gt = json.load(f)
    f.close()

    # convert det to the all_boxes list
    num_images = len(gt['images'])
    num_classes = 2
    print('Total number of images:',len(det['images']))
    all_boxes, all_segms, all_keyps = empty_results(num_classes,num_images)
    for cls in range(num_classes):
        for image in range(num_images):
            filename = gt['images'][image]['file_name']
            fid = gt['images'][image]['id']
            img_prop = get_by_filename(det,filename)
            if not (img_prop is None):
                img_id,det_prop = img_prop
                boxes = get_boxes_by_img_id(det,img_id)
                print('Reading detections for:',filename,'--',det_prop['file_name'])
                boxes = np.array([b['bbox'] for b in boxes])
                if len(boxes) > 0:
                    # add w, h to get (x2,y2)
                    boxes[:,2] += boxes[:,0]
                    boxes[:,3] += boxes[:,1]
                    all_boxes[cls][image] = boxes
            else:
                all_boxes[cls][image] = []
    # save detections
    with open(os.path.join(output_dir,'detections.pkl'),'wb') as f:
        pickle.dump(dict(all_boxes=all_boxes,all_segms=all_segms,all_keyps=all_keyps),f)
    f.close()
    #input(len(all_boxes[0]))
    coco_eval = evaluate_boxes(json_dataset,all_boxes,output_dir)
    #coco_eval = task_evaluation.evaluate_all(json_dataset,all_boxes,all_segms,all_keyps,output_dir)

    disp_detection_eval_metrics(json_dataset, coco_eval, iou_low=0.5, iou_high=0.5, output_dir=output_dir)
    disp_detection_eval_metrics(json_dataset, coco_eval, iou_low=0.75, iou_high=0.75, output_dir=output_dir)
    disp_detection_eval_metrics(json_dataset, coco_eval, iou_low=0.5, iou_high=0.95, output_dir=output_dir)


if __name__ == '__main__':
    eval_json(det_json,gt_dataset_name)
