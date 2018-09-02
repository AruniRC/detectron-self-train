#!/usr/bin/env bash


###     Faster R-CNN Resnet-50 detector trained on WIDER-Face
DET_NAME=frcnn-R-50-C4-1x
TRAIN_IMDB=WIDER
CFG_PATH=configs/wider_face/e2e_faster_rcnn_R-50-C4_1x.yaml
WT_PATH=Outputs/e2e_faster_rcnn_R-50-C4_1x/Jul30-15-51-27_node097_step/ckpt/model_step79999.pth
CONF_THRESH=0.25
OUT_DIR="Outputs/evaluations/"${DET_NAME}"/cs6/train-"${TRAIN_IMDB}"_train-video_conf-"${CONF_THRESH}


###     Run detector on all videos listed in "train" split of CS6
for VIDEO in `cat data/CS6/list_video_train.txt | sed -n "1,30p"`
do
    echo $VIDEO

    sbatch gypsum/scripts/eval/cs6/run_detector_full_video_1080.sbatch \
        ${VIDEO} \
        ${CFG_PATH} \
        ${WT_PATH} \
        ${OUT_DIR} \
        ${CONF_THRESH}

    # sbatch gypsum/scripts/eval/cs6/run_det_video.sbatch \
    #     $VIDEO \
    #     ${DET_NAME} \
    #     ${OUT_DIR} \
    #     ${CONF_THRESH}

done
