#!/usr/bin/env bash
#SBATCH --job-name=bdd_source_and_HP18k_noisy_070_domain_im_roi_cst
#SBATCH -o gypsum/logs/%j_bdd_source_and_HP18k_noisy_070_domain_im_roi_cst.txt 
#SBATCH -e gypsum/errs/%j_bdd_source_and_HP18k_noisy_070_domain_im_roi_cst.txt
#SBATCH -p 1080ti-long
#SBATCH --gres=gpu:4
#SBATCH --mem=60000
##SBATCH --cpus-per-task=4
##SBATCH --mem-per-cpu=4096



python tools/train_net_step.py \
    --dataset bdd_peds+bdd_HP18k_noisy_070 \
    --cfg configs/baselines/bdd_domain_im_roi_cst.yaml  \
    --set NUM_GPUS 1 TRAIN.SNAPSHOT_ITERS 5000 \
    --iter_size 2 \
    --use_tfboard \
    --load_ckpt /mnt/nfs/work1/elm/arunirc/Research/detectron-video/detectron_distill/Detectron-pytorch-video/Outputs/e2e_faster_rcnn_R-50-C4_1x/Jul30-15-51-27_node097_step/ckpt/model_step79999.pth \
    #--load_ckpt Outputs/cityscapes/baseline_bdd_clear_any_daytime/ckpt/model_step69999.pth \
    #--load_ckpt Outputs/e2e_faster_rcnn_R-50-C4_1x/Jul30-15-51-27_node097_step/ckpt/model_step79999.pth \

