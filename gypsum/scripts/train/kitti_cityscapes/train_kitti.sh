#!/usr/bin/env bash 
#SBATCH --job-name=train_kitti
#SBATCH -o gypsum/logs/%j_train_kitti.txt
#SBATCH -e gypsum/errs/%j_train_kitti.txt
#SBATCH -p titanx-long
#SBATCH --gres=gpu:4
#SBATCH --mem=100000
##SBATCH --cpus-per-task=4
##SBATCH --mem-per-cpu=4096

python tools/train_net_step.py \
    --dataset kitti_car_train \
    --cfg configs/baselines/kitti.yaml \
    --set TRAIN.SNAPSHOT_ITERS 10000 \
    --use_tfboard \
