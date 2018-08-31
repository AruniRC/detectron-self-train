
"""

Takes the FDDB/WIDER-style detections and saves the corresponding images under 
'data/CS6_annot/frames'. Frames are written to disk *only* if an image with the 
same name does not already exist in the folder. This ensures that if a frame is 
already written to disk for being part of CS6-GT, the CS6-dets will read it in, 
without wasting disk space on writing another copy of it.

File structure:
    data/CS6_annot/
        frames/<video-name>/*.jpg

Detections format:  [x y w h score]
            
This is done one video at a time, the video name being specified as an argument. 
This script can then be run in parallel by calling with different video names.

Usage:
    srun --pty --mem 10000 python tools/face/make_det_frames.py

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
sys.path.append('./tools')
import _init_paths
import numpy as np
import os, cv2
import argparse
import os.path as osp
import time
import skvideo.io
import json
import csv
from six.moves import xrange
import utils.face_utils as face_utils



VID_NAME = '3013.mp4'
DET_DIR = 'Outputs/evaluations/frcnn-R-50-C4-1x/cs6/train-WIDER_train-video_conf-0.25/'
DATA_DIR = 'data/CS6'
OUT_DIR = 'data/CS6_annot'

DEBUG = False

MIN_SZ = 16

def parse_args():
    parser = argparse.ArgumentParser(description='Creating CS6 ground truth data')
    parser.add_argument(
        '--output_dir', help='directory for saving outputs', default=OUT_DIR
    )
    parser.add_argument(
        '--data_dir', help='Path to CS6 data folder', default=DATA_DIR
    )
    parser.add_argument(
        '--det_dir', help='Path to CS6 data folder', default=DET_DIR
    )
    parser.add_argument(
        '--video_name', help='Name of video file', default=VID_NAME
    )
    return parser.parse_args()



_GREEN = (18, 127, 15)
# ------------------------------------------------------------------------------
def draw_detection_list(im, dets):
# ------------------------------------------------------------------------------
    """ Draw detected bounding boxes on a copy of image and return it.
        [x0 y0 w h conf_score]
    """
    im_det = im.copy()
    if dets.ndim == 1:
        dets = dets[np.newaxis,:] # handle single detection case

    # format into [xmin, ymin, xmax, ymax]
    dets[:, 2] = dets[:, 2] + dets[:, 0]
    dets[:, 3] = dets[:, 3] + dets[:, 1]

    for i, det in enumerate(dets):
        bbox = dets[i, :4]
        x0, y0, x1, y1 = [int(x) for x in bbox]
        line_color = _GREEN
        cv2.rectangle(im_det, (x0, y0), (x1, y1), line_color, thickness=2)

    return im_det



def filter_video_det(det_dict, video_name):
    '''
        Keep detections that correspond to "video_name"
        Returns: np.array([x y w h score frame_num], ...)
    '''
    annots = []

    video_frame_names = [x for x in det_dict.keys() if video_name+'_' in x]

    for im_name in video_frame_names:
        dets = det_dict[im_name]
        frame_num = int(im_name.split('_')[1])
        for d in dets:
            if d[2] <  MIN_SZ or d[3] < MIN_SZ:
                continue       
            annots.append( d + [frame_num] )
        print(im_name)

    annots = np.array(annots)
    assert annots.ndim == 2
    order =  np.argsort(annots[:,-1]) # FIXED: order frame-numbers (501.mp4 is shuffled)
    annots_ordered = annots[order,:]
    return annots_ordered


if __name__ == '__main__':
    

    args = parse_args()


    # --------------------------------------------------------------------------
    # Data setup
    # --------------------------------------------------------------------------

    # Video file
    video_path = osp.join(args.data_dir, 'videos', args.video_name)
    if osp.exists(video_path):
        videogen = skvideo.io.vreader(video_path)
    else:
        raise IOError('Path to video not found: \n%s' % video_path)


    # Outputs
    vid_name = osp.basename(video_path).split('.')[0]
    img_output_dir = osp.join(args.output_dir, 'frames', vid_name)
    if DEBUG:
        img_output_dir += '_debug-viz'

    if not osp.exists(img_output_dir):
        os.makedirs(img_output_dir)


    # Load pre-computed detections for that video
    det_file = osp.join(args.det_dir, vid_name + '.txt')
    det_dict = face_utils.parse_wider_gt(det_file)

    annots = filter_video_det(det_dict, vid_name)


    first_frame = int(annots[0,-1])
    last_frame = int(annots[-1,-1])
    assert last_frame > first_frame, \
        'Last frame (%d) cannot be smaller than first frame (%d)!' % (last_frame, 
                                                                      first_frame)

    
    # --------------------------------------------------------------------------
    # Saving video frames that have detections (and not pre-existing on disk)
    # --------------------------------------------------------------------------
    print('\nVIDEO_NAME: %s' % vid_name)
    print('video annotation size: ')
    print(annots.shape) # fix

    start = time.time()

    # NOTE: Frame number starts from ZERO
    for frame_num, im in enumerate(videogen):
        if frame_num < first_frame:
            continue
        if frame_num > last_frame:
            break

        sel = (annots[:,-1] == frame_num)

        # if no face annotated for that frame
        if np.sum(sel) == 0:
            continue

        frame_annots = annots[sel,:]

        im_name = '%s_%08d' % (vid_name, frame_num)

        print('Frame: %d/%d' % (frame_num, last_frame))
        print(im_name)

        im = im[:,:,(2,1,0)] # RGB --> BGR
        imw = im.shape[1]
        imh = im.shape[0]


        # Saving frames as JPG images
        viz_out_path = osp.join(img_output_dir, im_name + '.jpg')


        # DO NOT overwrite an existing image!
        if osp.exists(viz_out_path):
            print('Image exists: %s' % viz_out_path)
            continue

        if DEBUG:
            # save image with drawn bounding-boxes
            im_det = draw_detection_list( im, frame_annots[:,:4].copy() )
            cv2.imwrite(viz_out_path, im_det)
        else:
            # save just the image
            cv2.imwrite(viz_out_path, im)

        
        # if ((i + 1) % 100) == 0:
        #     sys.stdout.write('%d ' % i)
        #     sys.stdout.flush()

    end = time.time()
    print('Execution time in seconds: %f' % (end - start))
    print('Finished: SUCCESS')
        
        
        
