#!/usr/bin/env bash
#SBATCH --job-name=bdd_source_and_labeled_005
#SBATCH -o gypsum/logs/%j_bdd_source_and_labeled_005.txt 
#SBATCH -e gypsum/errs/%j_bdd_source_and_labeled_005.txt
#SBATCH -p 1080ti-long
#SBATCH --gres=gpu:8
#SBATCH --mem=100000
##SBATCH --cpus-per-task=4
##SBATCH --mem-per-cpu=4096



python tools/train_net_step.py \
    --dataset bdd_peds+labeled_005 \
    --cfg configs/baselines/bdd_peds_dets_bs64_4gpu.yaml  \
    --set NUM_GPUS 1 TRAIN.SNAPSHOT_ITERS 5000 \
    --iter_size 2 \
    --use_tfboard \
    --load_ckpt /mnt/nfs/work1/elm/arunirc/Research/detectron-video/detectron_distill/Detectron-pytorch-video/Outputs/e2e_faster_rcnn_R-50-C4_1x/Jul30-15-51-27_node097_step/ckpt/model_step79999.pth \
    #--load_ckpt /mnt/nfs/scratch1/pchakrabarty/ped_models/bdd_peds.pt \

