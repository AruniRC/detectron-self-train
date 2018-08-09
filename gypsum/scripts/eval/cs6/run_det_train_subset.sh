#!/usr/bin/env bash

# Run detector on a subset of CS6 train-set videos
# Usage: ./gypsum/scripts/eval/cs6/run_det_train_subset.sh


DET_NAME=frcnn-R-50-C4-1x
CONF_THRESH=0.25
# CFG_PATH=configs/wider_face/e2e_faster_rcnn_R-50-C4_1x.yaml
# WT_PATH=Outputs/e2e_faster_rcnn_R-50-C4_1x/Jul30-15-51-27_node097_step/ckpt/model_step79999.pth
OUT_DIR=Outputs/evaluations/${DET_NAME}/cs6/baseline_train_conf-${CONF_THRESH}



mkdir -p "Outputs/cache/face"

cat data/CS6/list_video_train.txt | \
    sort -R --random-source=data/CS6/list_video_val.txt | \
    tail -n 20 \
    > Outputs/cache/face/list_video_train_subset.txt



for VIDEO in `cat Outputs/cache/face/list_video_train_subset.txt`
do
    echo $VIDEO

    sbatch gypsum/scripts/eval/cs6/run_det_video.sbatch \
        $VIDEO \
        ${DET_NAME} \
        ${OUT_DIR} \
        ${CONF_THRESH}

done
