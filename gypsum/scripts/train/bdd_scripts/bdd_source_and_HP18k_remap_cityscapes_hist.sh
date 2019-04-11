#!/usr/bin/env bash
#SBATCH --job-name=bdd_source_and_HP18k_remap_cityscapes_hist
#SBATCH -o gypsum/logs/%j_bdd_source_and_HP18k_remap_cityscapes_hist.txt 
#SBATCH -e gypsum/errs/%j_bdd_source_and_HP18k_remap_cityscapes_hist.txt
#SBATCH -p 1080ti-long
#SBATCH --gres=gpu:4
#SBATCH --mem=100000
##SBATCH --cpus-per-task=4
##SBATCH --mem-per-cpu=4096



export CUDA_VISIBLE_DEVICES=3

python tools/train_net_step.py \
    --dataset bdd_peds+HP18k_remap_cityscape_hist \
    --cfg configs/baselines/bdd_distill100.yaml \
    --set NUM_GPUS 1 TRAIN.SNAPSHOT_ITERS 5000 \
    --iter_size 2 \
    --use_tfboard \
    --load_ckpt /mnt/nfs/scratch1/pchakrabarty/bdd_recs/ped_models/bdd_peds.pth \

