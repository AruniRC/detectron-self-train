
"""

Format:

    data/CS6_annot/
        frames/<video-name>/*.jpg
        video_annots/<video-name>.txt
            




"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
sys.path.append('./tools')
import numpy as np
import os, cv2
import argparse
import os.path as osp
import time
import skvideo.io
import json
import csv
from six.moves import xrange



VID_NAME = '501.mp4'
DATA_DIR = '/mnt/nfs/scratch1/arunirc/data/CS6/CS6/CS6.0.01/CS6'
OUT_DIR = '/mnt/nfs/work1/elm/arunirc/Data/CS6_annots'

DEBUG = False

def parse_args():
    parser = argparse.ArgumentParser(description='Creating CS6 ground truth data')
    parser.add_argument(
        '--output-dir', dest='output_dir', help='directory for saving outputs',
        default=OUT_DIR, type=str
    )
    parser.add_argument(
        '--data_dir', help='Path to CS6 data folder', default=DATA_DIR
    )
    parser.add_argument(
        '--video_name', help='Name of video file', default=VID_NAME
    )
    return parser.parse_args()



_GREEN = (18, 127, 15)
# -----------------------------------------------------------------------------------
def draw_detection_list(im, dets):
# -----------------------------------------------------------------------------------
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



def filter_video_gt(filename, video_name):
    '''
        Keep ground-truth CSV rows that correspond to "video_name"
    '''
    annots = []
    with open(filename, "r") as csvfile:
        datareader = csv.reader(csvfile)
        next(datareader, None)  # skip the headers
        count = 0
        for row in datareader:
            if video_name in row[0]:
                count += 1
                annots.append( [int(x) for x in row[1:]] ) 
                # [ x, y, w, h, frame_num ]
            else:
                continue
    annots = np.array(annots)
    order =  np.argsort(annots[:,-1]) # FIXED: order frame-numbers (501.mp4 is shuffled)
    annots_ordered = annots[order,:]
    return annots_ordered


if __name__ == '__main__':
    

    args = parse_args()


    # -----------------------------------------------------------------------------------
    # Data setup
    # -----------------------------------------------------------------------------------

    # video file
    video_path = osp.join(args.data_dir, 'videos', args.video_name)
    if osp.exists(video_path):
        videogen = skvideo.io.vreader(video_path)
    else:
        raise IOError('Path to video not found: \n%s' % video_path)


    # outputs
    vid_name = osp.basename(video_path).split('.')[0]
    img_output_dir = osp.join(args.output_dir, 'frames', vid_name)
    if DEBUG:
        img_output_dir += '_debug-viz'

    if not osp.exists(img_output_dir):
        os.makedirs(img_output_dir)

    annot_output_dir = osp.join(args.output_dir, 'video_annots') # txt files for each video
    if not osp.exists(annot_output_dir):
        os.makedirs(annot_output_dir)

    print(annot_output_dir)


    # ground truth
    gt_file = osp.join(args.data_dir, 'protocols', 
                       'cs6_face_detection_ground_truth.csv')
    annots = filter_video_gt(gt_file, vid_name)
    first_frame = annots[0,-1]
    last_frame = annots[-1,-1]
    assert last_frame > first_frame, 
        'Last frame (%d) cannot be smaller than first frame (%d)!' % (last_frame, first_frame)

    
    # -----------------------------------------------------------------------------------
    # Creating ground-truth from video frames
    # -----------------------------------------------------------------------------------
    print('\nVIDEO_NAME: %s' % vid_name)
    print('video annotation size: ')
    print(annots.shape) # fix

    start = time.time()
    with open(os.path.join(annot_output_dir, vid_name + '.txt'), 'w') as fid:
        det_list = []

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
            if DEBUG:
                # save image with drawn bounding-boxes
                im_det = draw_detection_list( im, frame_annots[:,:4].copy() )
                cv2.imwrite(viz_out_path, im_det)
            else:
                cv2.imwrite(viz_out_path, im)

            # Writing annotations to text file
            fid.write('frames/' + vid_name + '/' + im_name + '.jpg\n')
            fid.write(str(frame_annots.shape[0]) + '\n')

            for j in xrange(frame_annots.shape[0]):
                # make valid bounding boxes
                x1 = min(max(frame_annots[j, 0], 0), imw - 1)
                y1 = min(max(frame_annots[j, 1], 0), imh - 1)
                x2 = min(max(x1 + frame_annots[j, 2] - 1, 0), imw - 1)
                y2 = min(max(y1 + frame_annots[j, 3] - 1, 0), imh - 1)
                fid.write('%f %f %f %f\n' % (x1, y1, x2 - x1 + 1, y2 - y1 + 1) )

            
            # if ((i + 1) % 100) == 0:
            #     sys.stdout.write('%d ' % i)
            #     sys.stdout.flush()

    end = time.time()
    print('Execution time in seconds: %f' % (end - start))
    print('Finished: SUCCESS')
        
        
        
