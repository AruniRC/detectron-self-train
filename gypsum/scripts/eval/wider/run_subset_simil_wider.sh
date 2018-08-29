#!/bin/bash


GENRE='simil-WIDER'
VID_FOLDER='/mnt/nfs/scratch1/souyoungjin/face_detection_videos_similar_WIDER'
# MODEL_NAME='WIDER-subset-0.5'
# PROTO_FILE='models/face-hard-neg/VGG16/faster_rcnn_end2end_ohem/test.prototxt'
# WT_FILE='output/faster_rcnn_end2end/wider_subset_0.5/vgg16_faster_rcnn_face_iter_80000.caffemodel'

MODEL_NAME='frcnn-R-50-C4-1x'
CFG_PATH='configs/wider_face/e2e_faster_rcnn_R-50-C4_1x.yaml'
WT_PATH='Outputs/e2e_faster_rcnn_R-50-C4_1x/Jul30-15-51-27_node097_step/ckpt/model_step79999.pth'



for f in `ls ${VID_FOLDER}`
do
    echo 'Genre: '$GENRE'  Video folder: '${VID_FOLDER}'  video_name: '$f
    sbatch gypsum/scripts/eval/wider/run_vid.sbatch $GENRE ${VID_FOLDER} $f \
        ${MODEL_NAME} ${CFG_PATH} ${WT_PATH}
done



# time ./tools/domain-shift/covar_video.py \
#     --genre ${GENRE} \
#     --video_name ${VID_NAME}


# pip install pyyaml --user
# pip install matplotlib --user
# pip install opencv-python --user
# pip install sk-video --user




