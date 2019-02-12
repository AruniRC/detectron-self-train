#!/usr/bin/env bash
#SBATCH --job-name=eval_bdd_peds_joint_bs64_4gpu_on_bdd_peds_val
#SBATCH -o gypsum/logs/%j_eval_bdd_peds_joint_bs64_4gpu_peds_on_bdd_peds_val.txt 
#SBATCH -e gypsum/errs/%j_eval_bdd_peds_joint_bs64_4gpu_on_bdd_peds_val.txt
#SBATCH -p 1080ti-long
#SBATCH --gres=gpu:4
#SBATCH --mem=100000
##SBATCH --cpus-per-task=4
##SBATCH --mem-per-cpu=4096


# --load_ckpt Outputs/bdd_peds_dets_joint/bdd_peds_and_dets20k/ckpt/model_step14999.pth \
python tools/test_net.py \
    --dataset bdd_peds_full_val \
    --cfg configs/baselines/bdd_distill070.yaml \
    --output_dir results/bdd_source_and_HP18k_distill070/bdd_peds_full_val \
    --load_ckpt Outputs/bdd_distill070/bdd_source_and_HP18k_distill070/ckpt/model_step14999.pth \
    --set TEST.SCORE_THRESH 0.1 TRAIN.GT_SCORES False TRAIN.JOINT_TRAINING False\
    --multi-gpu-testing

# -- debugging --
# python tools/train_net_step.py \
#     --dataset cs6-train-easy-gt-sub+WIDER \
#     --cfg configs/cs6/e2e_faster_rcnn_R-50-C4_1x_quick.yaml  \
#     --load_ckpt Outputs/e2e_faster_rcnn_R-50-C4_1x/Jul30-15-51-27_node097_step/ckpt/model_step79999.pth \
#     --iter_size 2 \
#     --use_tfboard
