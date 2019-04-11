#!/usr/bin/env bash
#SBATCH --job-name=bdd_source_and_HP18k_noisy_080_domain_im_roi_cst
#SBATCH -o gypsum/logs/%j_bdd_source_and_HP18k_noisy_080_domain_im_roi_cst.txt 
#SBATCH -e gypsum/errs/%j_bdd_source_and_HP18k_noisy_080_domain_im_roi_cst.txt
#SBATCH -p 1080ti-long
#SBATCH --gres=gpu:4
#SBATCH --mem=60000
##SBATCH --cpus-per-task=4
##SBATCH --mem-per-cpu=4096



python tools/train_net_step.py \
    --dataset bdd_peds+bdd_HP18k_noisy_080 \
    --cfg configs/baselines/bdd_domain_im_roi_cst.yaml  \
    --set NUM_GPUS 1 TRAIN.SNAPSHOT_ITERS 5000 \
    --iter_size 2 \
    --use_tfboard \
    --load_ckpt /mnt/nfs/scratch1/pchakrabarty/bdd_recs/ped_models/bdd_peds.pth \


