#!/usr/bin/env bash


###     Faster R-CNN Resnet-50 detector trained on CS6-train-subset, LR=0.0001
# DET_NAME=frcnn-R-50-C4-1x-8gpu-lr=0.0001
# TRAIN_IMDB=cs6-subset
# CFG_PATH=configs/cs6/e2e_faster_rcnn_R-50-C4_1x_8gpu_lr=0.0001.yaml
# WT_PATH=Outputs/e2e_faster_rcnn_R-50-C4_1x_8gpu_lr=0.0001/Aug11-12-52-16_node151_step/ckpt/model_step29999.pth
# CONF_THRESH=0.25
# OUT_DIR="Outputs/evaluations/"${DET_NAME}"/cs6/train-"${TRAIN_IMDB}"_val-video_conf-"${CONF_THRESH}


###     Faster R-CNN Resnet-50 detector trained on CS6-train-subset, GT, LR=0.001
# DET_NAME=frcnn-R-50-C4-1x-8gpu
# TRAIN_IMDB=cs6-subset-GT
# CFG_PATH=configs/cs6/e2e_faster_rcnn_R-50-C4_1x_8gpu.yaml
# WT_PATH=Outputs/e2e_faster_rcnn_R-50-C4_1x_8gpu/Aug15-17-16-22_node105_step/ckpt/model_step29999.pth
# CONF_THRESH=0.25
# OUT_DIR="Outputs/evaluations/"${DET_NAME}"/cs6/train-"${TRAIN_IMDB}"_val-video_conf-"${CONF_THRESH}



###     Faster R-CNN Resnet-50 detector trained on CS6-train-subset, GT, LR=0.0001
# DET_NAME=frcnn-R-50-C4-1x-8gpu-lr=0.0001
# TRAIN_IMDB=cs6-subset-GT
# CFG_PATH=configs/cs6/e2e_faster_rcnn_R-50-C4_1x_8gpu_lr=0.0001.yaml
# WT_PATH=Outputs/e2e_faster_rcnn_R-50-C4_1x_8gpu_lr=0.0001/Aug15-22-04-05_node140_step/ckpt/model_step29999.pth
# CONF_THRESH=0.25
# OUT_DIR="Outputs/evaluations/"${DET_NAME}"/cs6/train-"${TRAIN_IMDB}"_val-video_conf-"${CONF_THRESH}


###     Faster R-CNN Resnet-50 detector trained on CS6-train-subset + WIDER
# DET_NAME=frcnn-R-50-C4-1x-8gpu-lr=0.0001
# TRAIN_IMDB=cs6-subset+WIDER
# CFG_PATH=configs/cs6/e2e_faster_rcnn_R-50-C4_1x_8gpu_lr=0.0001.yaml
# WT_PATH=Outputs/e2e_faster_rcnn_R-50-C4_1x_8gpu_lr=0.0001_cs6_WIDER/Aug15-22-45-30_node141_step/ckpt/model_step49999.pth
# CONF_THRESH=0.25
# OUT_DIR="Outputs/evaluations/"${DET_NAME}"/cs6/train-"${TRAIN_IMDB}"_val-video_conf-"${CONF_THRESH}


# ###     Faster R-CNN Resnet-50 detector trained on CS6-train-subset-GT + WIDER
# DET_NAME=frcnn-R-50-C4-1x-8gpu-lr=0.0001
# TRAIN_IMDB=cs6-subset-GT+WIDER
# CFG_PATH=configs/cs6/e2e_faster_rcnn_R-50-C4_1x_8gpu_lr=0.0001.yaml
# WT_PATH=Outputs/e2e_faster_rcnn_R-50-C4_1x_8gpu_lr=0.0001_cs6_WIDER/Aug15-22-45-51_node142_step/ckpt/model_step49999.pth
# CONF_THRESH=0.25
# OUT_DIR="Outputs/evaluations/"${DET_NAME}"/cs6/train-"${TRAIN_IMDB}"_val-video_conf-"${CONF_THRESH}


###########     Training on *FULL* CS6-GT 

# ###     Faster R-CNN Resnet-50 detector trained on intermediate checkpoint of CS6-train-GT
# DET_NAME=frcnn-R-50-C4-1x-8gpu-100k
# TRAIN_IMDB=cs6-GT-chkpt-30k
# CFG_PATH=configs/cs6/e2e_faster_rcnn_R-50-C4_1x_8gpu_100k.yaml
# WT_PATH=Outputs/e2e_faster_rcnn_R-50-C4_1x_8gpu_100k/Aug26-18-18-43_node111_step/ckpt/model_step29999.pth

# ##
# DET_NAME=frcnn-R-50-C4-1x-8gpu-100k
# TRAIN_IMDB=cs6-GT-chkpt-60k
# CFG_PATH=configs/cs6/e2e_faster_rcnn_R-50-C4_1x_8gpu_100k.yaml
# WT_PATH=Outputs/e2e_faster_rcnn_R-50-C4_1x_8gpu_100k/Aug26-18-18-43_node111_step/ckpt/model_step59999.pth

# ##
# DET_NAME=frcnn-R-50-C4-1x-8gpu-100k
# TRAIN_IMDB=cs6-GT-chkpt-100k
# CFG_PATH=configs/cs6/e2e_faster_rcnn_R-50-C4_1x_8gpu_100k.yaml
# WT_PATH=Outputs/e2e_faster_rcnn_R-50-C4_1x_8gpu_100k/Aug26-18-18-43_node111_step/ckpt/model_step99999.pth

# # ##
# DET_NAME=frcnn-R-50-C4-1x-8gpu-100k
# TRAIN_IMDB=cs6-GT-chkpt-10k
# CFG_PATH=configs/cs6/e2e_faster_rcnn_R-50-C4_1x_8gpu_100k.yaml
# WT_PATH=Outputs/e2e_faster_rcnn_R-50-C4_1x_8gpu_100k/Aug26-18-18-43_node111_step/ckpt/model_step9999.pth




##         Fine-tuned on CS6-Train-GT
# DET_NAME=frcnn-R-50-C4-1x-8gpu-100k-lr=0.0001
# TRAIN_IMDB=cs6-GT-chkpt-30k
# CFG_PATH=configs/cs6/e2e_faster_rcnn_R-50-C4_1x_8gpu_100k_lr=0.0001.yaml
# WT_PATH=Outputs/e2e_faster_rcnn_R-50-C4_1x_8gpu_100k_lr=0.0001/Aug26-18-28-46_node112_step/ckpt/model_step29999.pth

## 
# DET_NAME=frcnn-R-50-C4-1x-8gpu-100k-lr=0.0001
# TRAIN_IMDB=cs6-GT-chkpt-60k
# CFG_PATH=configs/cs6/e2e_faster_rcnn_R-50-C4_1x_8gpu_100k_lr=0.0001.yaml
# WT_PATH=Outputs/e2e_faster_rcnn_R-50-C4_1x_8gpu_100k_lr=0.0001/Aug26-18-28-46_node112_step/ckpt/model_step59999.pth

# ## 
# DET_NAME=frcnn-R-50-C4-1x-8gpu-100k-lr=0.0001
# TRAIN_IMDB=cs6-GT-chkpt-100k
# CFG_PATH=configs/cs6/e2e_faster_rcnn_R-50-C4_1x_8gpu_100k_lr=0.0001.yaml
# WT_PATH=Outputs/e2e_faster_rcnn_R-50-C4_1x_8gpu_100k_lr=0.0001/Aug26-18-28-46_node112_step/ckpt/model_step99999.pth

## 
# DET_NAME=frcnn-R-50-C4-1x-8gpu-100k-lr=0.0001
# TRAIN_IMDB=cs6-GT-chkpt-10k
# CFG_PATH=configs/cs6/e2e_faster_rcnn_R-50-C4_1x_8gpu_100k_lr=0.0001.yaml
# WT_PATH=Outputs/e2e_faster_rcnn_R-50-C4_1x_8gpu_100k_lr=0.0001/Aug26-18-28-46_node112_step/ckpt/model_step9999.pth


# DET_NAME=frcnn-R-50-C4-1x-8gpu-50k
# TRAIN_IMDB=cs6-train-dets
# CFG_PATH=configs/cs6/e2e_faster_rcnn_R-50-C4_1x_8gpu_50k.yaml
# WT_PATH=Outputs/e2e_faster_rcnn_R-50-C4_1x_8gpu_50k/Sep01-16-49-28_node141_step/ckpt/model_step49999.pth


# DET_NAME=frcnn-R-50-C4-1x
# TRAIN_IMDB=WIDER
# CFG_PATH=configs/wider_face/e2e_faster_rcnn_R-50-C4_1x.yaml
# WT_PATH=Outputs/e2e_faster_rcnn_R-50-C4_1x/Jul30-15-51-27_node097_step/ckpt/model_step79999.pth


DET_NAME=frcnn-R-50-C4-1x
TRAIN_IMDB=CS6-Dets-30k
CFG_PATH=configs/cs6/e2e_faster_rcnn_R-50-C4_1x_8gpu_30k.yaml
WT_PATH=Outputs/e2e_faster_rcnn_R-50-C4_1x_8gpu_30k/Sep15-00-00-47_node124_step/ckpt/model_step29999.pth


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
