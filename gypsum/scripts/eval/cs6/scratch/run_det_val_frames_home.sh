#!/usr/bin/env bash

# Run detector on the saved frames of CS6 "val" set
# Usage: ./gypsum/scripts/eval/cs6/run_det_val_frames.sh


# DET_NAME=frcnn-R-50-C4-1x_CS6-subset
# CONF_THRESH=0.25
# # CFG_PATH=configs/wider_face/e2e_faster_rcnn_R-50-C4_1x.yaml
# # WT_PATH=Outputs/e2e_faster_rcnn_R-50-C4_1x/Jul30-15-51-27_node097_step/ckpt/model_step79999.pth
# OUT_DIR=Outputs/evaluations/${DET_NAME}/cs6/baseline_train_conf-${CONF_THRESH}



# mkdir -p "Outputs/cache/face"

# cat data/CS6/list_video_train.txt | \
#     sort -R --random-source=data/CS6/list_video_val.txt | \
#     tail -n 20 \
#     > Outputs/cache/face/list_video_train_subset.txt



# ###     Faster R-CNN Resnet-50 detector trained on CS6-train-subset
# DET_NAME=frcnn-R-50-C4-1x
# TRAIN_IMDB=cs6-subset
# CFG_PATH=configs/cs6/e2e_faster_rcnn_R-50-C4_1x_8gpu.yaml
# WT_PATH=Outputs/e2e_faster_rcnn_R-50-C4_1x_8gpu/Aug09-21-48-42_node142_step/ckpt/model_step29999.pth
# CONF_THRESH=0.25
# OUT_DIR="Outputs/evaluations/"${DET_NAME}"/cs6/TRAIN="${TRAIN_IMDB}"_TEST=val_conf="${CONF_THRESH}


DET_NAME=frcnn-R-50-C4-1x
TRAIN_IMDB=cs6-subset-lr=0.0001
CFG_PATH=configs/cs6/e2e_faster_rcnn_R-50-C4_1x_8gpu.yaml
WT_PATH=Outputs/e2e_faster_rcnn_R-50-C4_1x_8gpu_lr=0.0001/Aug11-12-52-16_node151_step/ckpt/model_step29999.pth
CONF_THRESH=0.25
OUT_DIR="Outputs/evaluations/"${DET_NAME}"/cs6/HOME-FOLDER_TRAIN="${TRAIN_IMDB}"_TEST=val_conf="${CONF_THRESH}
DATA_DIR=/home/arunirc/Data/CS6_annot


VIDEO=801.mp4



###     Run detector on all videos listed in "val" split of CS6
# for VIDEO in `cat data/CS6/list_video_val.txt`
# do
echo $VIDEO

sbatch gypsum/scripts/eval/cs6/run_detector_frames.sbatch \
    ${VIDEO} \
    ${CFG_PATH} \
    ${WT_PATH} \
    ${OUT_DIR} \
    ${CONF_THRESH} \
    ${DATA_DIR}

    # sbatch gypsum/scripts/eval/cs6/run_det_video.sbatch \
    #     $VIDEO \
    #     ${DET_NAME} \
    #     ${OUT_DIR} \
    #     ${CONF_THRESH}

# done
