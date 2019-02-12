#!/usr/bin/env bash
#SBATCH --job-name=eval_cscapes_peds_on_bdd_peds_val
#SBATCH -o gypsum/logs/%j_eval_cscapes_peds_on_bdd_peds_val.txt 
#SBATCH -e gypsum/errs/%j_eval_cscapes_peds_on_bdd_peds_val.txt
#SBATCH -p 1080ti-long
#SBATCH --gres=gpu:4
#SBATCH --mem=100000
##SBATCH --cpus-per-task=4
##SBATCH --mem-per-cpu=4096


python tools/test_net.py \
    --dataset bdd_peds_full_val \
    --cfg configs/baselines/bdd_peds_dets_joint.yaml \
    --output_dir results/bdd_peds_and_dets20k_140k_iters/bdd_peds_val \
    --load_ckpt Outputs/bdd_peds_dets_joint/bdd_peds_and_dets20k_140k_iters/ckpt/model_step69999.pth \
    --set TEST.SCORE_THRESH 0.1 \
    --multi-gpu-testing

# -- debugging --
# python tools/train_net_step.py \
#     --dataset cs6-train-easy-gt-sub+WIDER \
#     --cfg configs/cs6/e2e_faster_rcnn_R-50-C4_1x_quick.yaml  \
#     --load_ckpt Outputs/e2e_faster_rcnn_R-50-C4_1x/Jul30-15-51-27_node097_step/ckpt/model_step79999.pth \
#     --iter_size 2 \
#     --use_tfboard
