#!/usr/bin/env bash
#SBATCH --job-name=D_train_wider
#SBATCH -o gypsum/logs/%j_train_wider.txt 
#SBATCH -e gypsum/errs/%j_train_wider.txt
#SBATCH -p titanx-long
#SBATCH --gres=gpu:4
#SBATCH --mem=100000
##SBATCH --cpus-per-task=4
##SBATCH --mem-per-cpu=4096


python tools/train_net_step.py \
    --dataset wider_train \
    --cfg configs/wider_face/e2e_faster_rcnn_R-50-C4_1x.yaml  \
    --use_tfboard

