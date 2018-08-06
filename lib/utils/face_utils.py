
from __future__ import division
import scipy.optimize
import numpy as np
import cv2


# -----------------------------------------------------------------------------------------
def parse_wider_gt(dets_file_name, isEllipse=False):
# -----------------------------------------------------------------------------------------
  '''
    Parse the FDDB-format detection output file:
      - first line is image file name
      - second line is an integer, for `n` detections in that image
      - next `n` lines are detection coordinates
      - again, next line is image file name
      - detections are [x y width height score]

    Returns a dict: {'img_filename': detections as a list of arrays}
  '''
  fid = open(dets_file_name, 'r')

  # Parsing the FDDB-format detection output txt file
  img_flag = True
  numdet_flag = False
  start_det_count = False
  det_count = 0
  numdet = -1

  det_dict = {}
  img_file = ''

  for line in fid:
    line = line.strip()

    if img_flag:
      # Image filename
      img_flag = False
      numdet_flag = True
      # print 'Img file: ' + line
      img_file = line
      det_dict[img_file] = [] # init detections list for image
      continue

    if numdet_flag:
      # next line after image filename: number of detections
      numdet = int(line)
      numdet_flag = False
      if numdet > 0:
        start_det_count = True # start counting detections
        det_count = 0
      else:
        # no detections in this image
        img_flag = True # next line is another image file
        numdet = -1

      # print 'num det: ' + line
      continue

    if start_det_count:
      # after numdet, lines are detections
      detection = [float(x) for x in line.split()] # split on whitespace
      det_dict[img_file].append(detection)
      # print 'Detection: %s' % line
      det_count += 1
      
    if det_count == numdet:
      start_det_count = False
      det_count = 0
      img_flag = True # next line is image file
      numdet_flag = False
      numdet = -1

  return det_dict


# -----------------------------------------------------------------------------------------
def crop_im_bbox(im_file, bbox, dilation_factor=0.0):
# -----------------------------------------------------------------------------------------
  '''
  Crop a bounding-box region out of an image.
  im_file: full path to image file 
  bbox format: [xmin ymin xmax ymax]
  '''
  im = cv2.imread(im_file)
  im = im[:, :, (2, 1, 0)]
  bbox_w = bbox[2]-bbox[0]
  bbox_h = bbox[3]-bbox[1]
  shift_x = bbox_w * dilation_factor
  shift_y = bbox_h * dilation_factor
  bbox_dilated = np.array([bbox[0] - shift_x, bbox[1] - shift_y, 
                  bbox[2] + shift_x, bbox[3] + shift_y])
  (x1,y1,x2,y2) = [ int(np.rint(x)) for x in bbox_dilated]
  # TODO - handle margin overflows
  return im[ y1:y2, x1:x2, : ]


# -----------------------------------------------------------------------------------------
def bbox_iou(boxA, boxB):
# -----------------------------------------------------------------------------------------
  # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
  
  # Determine the (x, y)-coordinates of the intersection rectangle
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])

  interW = xB - xA + 1
  interH = yB - yA + 1

  # Correction: reject non-overlapping boxes 
  # (NOTE: this will give -ve IoU)
  if interW <=0 or interH <=0 :
    return -1.0

  interArea = interW * interH
  boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
  boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
  iou = interArea / float(boxAArea + boxBArea - interArea)
  return iou


# -----------------------------------------------------------------------------------------
def bbox_iou_matrix(bbox_gt, bbox_pred):
# -----------------------------------------------------------------------------------------
  n_true = bbox_gt.shape[0]
  n_pred = bbox_pred.shape[0]

  # NUM_GT x NUM_PRED
  iou_matrix = np.zeros((n_true, n_pred))
  for i in range(n_true):
      for j in range(n_pred):
          iou_matrix[i, j] = bbox_iou(bbox_gt[i,:], bbox_pred[j,:])

  return iou_matrix


# -----------------------------------------------------------------------------------------
def match_bboxes(bbox_gt, bbox_pred, IOU_THRESH=0.5):
# -----------------------------------------------------------------------------------------
  '''
  Given sets of true and predicted bounding-boxes
  determine the best possible match.

  Parameters
  ----------
  bbox_gt, bbox_pred : N1x4 and N2x4 np array of bboxes [x1,y1,x2,y2]. 
    The number of bboxes, N1 and N2, need not be the same.
  
  Returns
  -------
  (idxs_true, idxs_pred, ious, label)
      idxs_true, idxs_pred : indices into gt and pred for matches
      ious : corresponding IOU value of each match
  '''
  n_true = bbox_gt.shape[0]
  n_pred = bbox_pred.shape[0]
  MAX_DIST = 1.0
  MIN_IOU = 0.0

  # NUM_GT x NUM_PRED
  iou_matrix = np.zeros((n_true, n_pred))
  for i in range(n_true):
      for j in range(n_pred):
          iou_matrix[i, j] = bbox_iou(bbox_gt[i,:], bbox_pred[j,:])

  if n_pred > n_true:
    # there are more predictions than ground-truth - add dummy rows
    diff = n_pred - n_true
    iou_matrix = np.concatenate( (iou_matrix, 
                                  np.full((diff, n_pred), MIN_IOU)), 
                                axis=0)

  if n_true > n_pred:
    # more ground-truth than predictions - add dummy columns
    diff = n_true - n_pred
    iou_matrix = np.concatenate( (iou_matrix, 
                                  np.full((n_true, diff), MIN_IOU)), 
                                axis=1)

  # call the Hungarian matching
  idxs_true, idxs_pred = scipy.optimize.linear_sum_assignment(1 - iou_matrix)

  if (not idxs_true.size) or (not idxs_pred.size):
      ious = np.array([])
  else:
      ious = iou_matrix[idxs_true, idxs_pred]

  # remove dummy assignments
  sel_pred = idxs_pred < n_pred
  idx_pred_actual = idxs_pred[sel_pred] 
  idx_gt_actual = idxs_true[sel_pred]
  ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
  sel_valid = (ious_actual > IOU_THRESH)
  label = sel_valid.astype(int)

  return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid], label 


# -----------------------------------------------------------------------------------------
def get_ellipse_rect(x, y, major, minor, angle_deg):
# -----------------------------------------------------------------------------------------
  '''
  Compute axis-aligned rectangle from FDDB ellipse annotation.
  '''
  x_values = [x + major * np.cos(t) * np.cos(np.radians(angle_deg)) -
                    minor * np.sin(t) * np.sin(np.radians(angle_deg)) \
                    for t in np.linspace(-np.pi, np.pi, 500)]

  y_values = [y + minor * np.sin(t) * np.cos(np.radians(angle_deg)) +
                    major * np.cos(t) * np.sin(np.radians(angle_deg)) \
                    for t in np.linspace(-np.pi, np.pi, 500)]

  return [np.min(x_values), np.min(y_values), 
          np.max(x_values), np.max(y_values)]


# -----------------------------------------------------------------------------------------
color_dict = {'red': (0,0,225), 'green': (0,255,0), 'yellow': (0,255,255), 'blue': (255,0,0), 
                '_GREEN':(18, 127, 15), '_GRAY': (218, 227, 218)}


# -----------------------------------------------------------------------------------------
def vis_bbox(img, bbox, thick=2, color='green'):
# -----------------------------------------------------------------------------------------
  """Visualizes a bounding box."""
  (x0, y0, x1, y1) = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
  cv2.rectangle(img, (x0, y0), (x1, y1), color_dict[color], 
                thickness=thick)
  return img

# -----------------------------------------------------------------------------------------
def _draw_string(img, pos, txt, font_scale=0.35):
# -----------------------------------------------------------------------------------------
  x0, y0 = int(pos[0]), int(pos[1])  
  # Compute text size.
  font = cv2.FONT_HERSHEY_SIMPLEX
  ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
  # Place text background.
  back_tl = x0, y0 - int(1.3 * txt_h)
  back_br = x0 + txt_w, y0
  cv2.rectangle(img, back_tl, back_br, color_dict['_GREEN'], -1)
  # Show text.
  txt_tl = x0, y0 - int(0.3 * txt_h)
  cv2.putText(img, txt, txt_tl, font, font_scale, 
              color_dict['_GRAY'], lineType=cv2.LINE_AA)

# -----------------------------------------------------------------------------------------
