#!/usr/bin/env bash
#SBATCH --job-name=distill_cs6
#SBATCH -o gypsum/logs/%j_distill_cs6.txt 
#SBATCH -e gypsum/errs/%j_distill_cs6.txt
#SBATCH -p 1080ti-long
#SBATCH --gres=gpu:8
#SBATCH --mem=200000
##SBATCH --cpus-per-task=4
##SBATCH --mem-per-cpu=4096



# Starts from WIDER pre-trained model, trains using pseudo-labels of CS6 detections 
# and the detection scores with distillation loss


python tools/train_net_step.py \
    --dataset cs6-train-det-score \
    --cfg configs/cs6/e2e_faster_rcnn_R-50-C4_1x_distill_0.8.yaml  \
    --load_ckpt Outputs/e2e_faster_rcnn_R-50-C4_1x/Jul30-15-51-27_node097_step/ckpt/model_step79999.pth \
    --use_tfboard

