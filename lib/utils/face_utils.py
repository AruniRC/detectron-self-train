
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

