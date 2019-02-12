#!/usr/bin/env bash
#SBATCH --job-name=da-im_cs6-HP-WIDER
#SBATCH -o gypsum/logs/%j_cs6-HP-WIDER.txt 
#SBATCH -e gypsum/errs/%j_cs6-HP-WIDER.txt
#SBATCH -p 1080ti-long
#SBATCH --gres=gpu:4
#SBATCH --mem=60000
##SBATCH --cpus-per-task=4
##SBATCH --mem-per-cpu=4096



# starts from WIDER pre-trained model
# trial run: just using CS6 data (imdb merging not done)


python tools/train_net_step.py \
    --dataset cityscapes_cars_HPlen3+kitti_car_train \
    --cfg configs/baselines/kitti_HP.yaml  \
    --set NUM_GPUS 1 TRAIN.SNAPSHOT_ITERS 5000 \
    --iter_size 2 \
    --use_tfboard \
    --load_ckpt /mnt/nfs/work1/elm/arunirc/Research/detectron-video/detectron_distill/Detectron-pytorch-video/Outputs/e2e_faster_rcnn_R-50-C4_1x/Dec05-21-53-14_node093_step/ckpt/model_step14999.pth \


