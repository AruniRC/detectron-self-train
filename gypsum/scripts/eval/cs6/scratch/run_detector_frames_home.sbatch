#!/usr/bin/env bash
#SBATCH --job-name=cs6_frames
#SBATCH -o gypsum/logs/%j_cs6_frames.txt 
#SBATCH -e gypsum/errs/%j_cs6_frames.txt
#SBATCH -p titanx-long
#SBATCH --gres=gpu:1
#SBATCH --mem=80000
##SBATCH --cpus-per-task=4
##SBATCH --mem-per-cpu=4096

# Run detector on a CS6 frames
# Usage: TODO

VIDNAME="$1"
CFG_PATH="$2"
WT_PATH="$3"
OUT_DIR="$4"
CONF_THRESH="$5"
DATA_DIR="$6"


python tools/face/detect_frames.py \
    --video_name ${VIDNAME} \
    --cfg ${CFG_PATH} \
    --load_ckpt ${WT_PATH} \
    --output-dir ${OUT_DIR} \
    --thresh ${CONF_THRESH} \
    --data_dir ${DATA_DIR}


# python tools/face/detect_video.py \
#     --video_name $1 \
#     --exp_name $2 \
#     --output-dir $3 \
#     --thresh $4 \
#     --vis



