#!/usr/bin/env bash
#SBATCH --job-name=Dft_cs6
#SBATCH -o gypsum/logs/%j_finetune_cs6.txt 
#SBATCH -e gypsum/errs/%j_finetune_cs6.txt
#SBATCH -p 1080ti-long
#SBATCH --gres=gpu:8
#SBATCH --mem=200000
##SBATCH --cpus-per-task=4
##SBATCH --mem-per-cpu=4096



# starts from WIDER pre-trained model
# trial run: just using CS6 "train-subset" data (imdb merging not done)
# NOTE: Uses lower learning rate than "train_cs6_basic.sh"


python tools/train_net_step.py \
    --dataset cs6-subset-score \
    --cfg configs/cs6/e2e_faster_rcnn_R-50-C4_1x_distill-scratch-baseline.yaml  \
    --load_ckpt Outputs/e2e_faster_rcnn_R-50-C4_1x/Jul30-15-51-27_node097_step/ckpt/model_step79999.pth \
    --use_tfboard

