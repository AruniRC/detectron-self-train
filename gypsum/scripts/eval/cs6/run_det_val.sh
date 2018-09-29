#!/usr/bin/env bash


## --- CS6-GT-subset ---
# DET_NAME=frcnn-R-50-C4-1x
# TRAIN_IMDB=CS6-GT-sub-50k
# CFG_PATH=configs/cs6/e2e_faster_rcnn_R-50-C4_1x_8gpu_50k.yaml
# WT_PATH=Outputs/e2e_faster_rcnn_R-50-C4_1x_8gpu_50k/Sep15-09-38-08_node121_step/ckpt/model_step49999.pth



## --- CS6-GT All videos ---
# DET_NAME=frcnn-R-50-C4-1x
# TRAIN_IMDB=CS6-GT-all-30k
# CFG_PATH=configs/cs6/e2e_faster_rcnn_R-50-C4_1x_8gpu_100k.yaml
# WT_PATH=Outputs/e2e_faster_rcnn_R-50-C4_1x_8gpu_100k/Aug26-18-18-43_node111_step/ckpt/model_step29999.pth

# DET_NAME=frcnn-R-50-C4-1x
# TRAIN_IMDB=CS6-GT-all-10k
# CFG_PATH=configs/cs6/e2e_faster_rcnn_R-50-C4_1x_8gpu_100k.yaml
# WT_PATH=Outputs/e2e_faster_rcnn_R-50-C4_1x_8gpu_100k/Aug26-18-18-43_node111_step/ckpt/model_step9999.pth


## --- CS6-Det videos ---
# DET_NAME=frcnn-R-50-C4-1x
# TRAIN_IMDB=CS6-Det-all-30k
# CFG_PATH=configs/cs6/e2e_faster_rcnn_R-50-C4_1x_8gpu_50k.yaml
# WT_PATH=Outputs/e2e_faster_rcnn_R-50-C4_1x_8gpu_50k/Sep06-16-42-46_node122_step/ckpt/model_step29999.pth




## --- CS6 + WIDER Joint training [distill branch] ---
DET_NAME=frcnn-R-50-C4-1x
TRAIN_IMDB=CS6-train-HP+WIDER-bs512-10k
CFG_PATH=configs/cs6/e2e_faster_rcnn_R-50-C4_1x_8gpu_joint-baseline_30k.yaml
WT_PATH=/mnt/nfs/work1/elm/arunirc/Research/detectron-video/detectron_distill/Detectron-pytorch-video/Outputs/e2e_faster_rcnn_R-50-C4_1x_8gpu_joint-baseline_30k/Sep20-23-01-54_node109_step/ckpt/model_step9999.pth


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
