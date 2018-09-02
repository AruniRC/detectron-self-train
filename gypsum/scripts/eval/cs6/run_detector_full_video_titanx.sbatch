#!/usr/bin/env bash
#SBATCH --job-name=cs6_video
#SBATCH -o gypsum/logs/%j_cs6_video.txt 
#SBATCH -e gypsum/errs/%j_cs6_video.txt
#SBATCH -p titanx-long
#SBATCH --gres=gpu:1
#SBATCH --mem=80000
##SBATCH --cpus-per-task=4
##SBATCH --mem-per-cpu=4096

# Run detector on a CS6 video
# Usage: call this from "run_det_val.sh"

VIDNAME="$1"
CFG_PATH="$2"
WT_PATH="$3"
OUT_DIR="$4"
CONF_THRESH="$5"


python tools/face/detect_video.py \
    --video_name ${VIDNAME} \
    --cfg ${CFG_PATH} \
    --load_ckpt ${WT_PATH} \
    --output-dir ${OUT_DIR} \
    --thresh ${CONF_THRESH}




# python tools/face/detect_video.py \
#     --video_name $1 \
#     --exp_name $2 \
#     --output-dir $3 \
#     --thresh $4 \
#     --vis



