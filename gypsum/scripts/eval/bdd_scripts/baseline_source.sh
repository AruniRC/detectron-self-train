#!/usr/bin/env bash
#SBATCH --job-name=basleine_source_eval
#SBATCH -o gypsum/logs/%j_baseline_source_eval.txt 
#SBATCH -e gypsum/errs/%j_baseline_source_eval.txt
#SBATCH -p 1080ti-short
#SBATCH --gres=gpu:8
#SBATCH --mem=100000
##SBATCH --cpus-per-task=4
##SBATCH --mem-per-cpu=4096

python tools/test_net.py \
    --set TEST.SCORE_THRESH 0.1 TRAIN.JOINT_TRAINING False TRAIN.GT_SCORES False \
    --dataset bdd_peds_val \
    --cfg configs/baselines/bdd100k.yaml \
    --load_ckpt <path_to_ckpt_file> \
    --multi-gpu-testing \

