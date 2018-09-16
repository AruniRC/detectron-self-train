#!/usr/bin/env bash


###     Faster R-CNN Resnet-50 detector trained on CS6-train-subset, LR=0.0001
# DET_NAME=frcnn-R-50-C4-1x-8gpu-lr=0.0001
# TRAIN_IMDB=cs6-subset
# CFG_PATH=configs/cs6/e2e_faster_rcnn_R-50-C4_1x_8gpu_lr=0.0001.yaml
# WT_PATH=Outputs/e2e_faster_rcnn_R-50-C4_1x_8gpu_lr=0.0001/Aug11-12-52-16_node151_step/ckpt/model_step29999.pth
# CONF_THRESH=0.25
# OUT_DIR="Outputs/evaluations/"${DET_NAME}"/cs6/train-"${TRAIN_IMDB}"_val-video_conf-"${CONF_THRESH}

## 
# DET_NAME=frcnn-R-50-C4-1x-8gpu-100k-lr=0.0001
# TRAIN_IMDB=cs6-GT-chkpt-10k
# CFG_PATH=configs/cs6/e2e_faster_rcnn_R-50-C4_1x_8gpu_100k_lr=0.0001.yaml
# WT_PATH=Outputs/e2e_faster_rcnn_R-50-C4_1x_8gpu_100k_lr=0.0001/Aug26-18-28-46_node112_step/ckpt/model_step9999.pth

# DET_NAME=frcnn-R-50-C4-1x
# TRAIN_IMDB=WIDER
# CFG_PATH=configs/wider_face/e2e_faster_rcnn_R-50-C4_1x.yaml
# WT_PATH=Outputs/e2e_faster_rcnn_R-50-C4_1x/Jul30-15-51-27_node097_step/ckpt/model_step79999.pth


DET_NAME=frcnn-R-50-C4-1x
TRAIN_IMDB=CS6-Dets-50k
CFG_PATH=configs/cs6/e2e_faster_rcnn_R-50-C4_1x_8gpu_50k.yaml
WT_PATH=Outputs/e2e_faster_rcnn_R-50-C4_1x_8gpu_50k/Sep15-09-38-08_node121_step/ckpt/model_step49999.pth


CONF_THRESH=0.1
OUT_DIR="Outputs/evaluations/"${DET_NAME}"/cs6/train-"${TRAIN_IMDB}"_val-easy_conf-"${CONF_THRESH}



###     Run detector on all videos listed in "val easy" split of CS6
for VIDEO in `cat data/CS6/list_video_val_easy.txt`
do
    echo $VIDEO

    sbatch gypsum/scripts/eval/cs6/run_detector_full_video_titanx.sbatch \
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
