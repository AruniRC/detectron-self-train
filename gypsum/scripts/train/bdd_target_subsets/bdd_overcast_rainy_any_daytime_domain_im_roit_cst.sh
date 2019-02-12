#!/usr/bin/env bash
#SBATCH --job-name=da-cst-im-roi_cs6-HP-WIDER
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
    --dataset bdd_peds+HP18k_overcast,rainy_any_daytime \
    --cfg configs/baselines/bdd_domain_im_roi_cst.yaml  \
    --set NUM_GPUS 1 TRAIN.SNAPSHOT_ITERS 5000 \
    --iter_size 2 \
    --use_tfboard \
    #--load_ckpt /mnt/nfs/scratch1/pchakrabarty/ped_models/bdd_peds.pth \
    #--load_ckpt Outputs/e2e_faster_rcnn_R-50-C4_1x/Jul30-15-51-27_node097_step/ckpt/model_step79999.pth \


# -- debugging --
# python tools/train_net_step.py \
#     --dataset cs6-train-easy-gt-sub+WIDER \
#     --cfg configs/cs6/e2e_faster_rcnn_R-50-C4_1x_quick.yaml  \
#     --load_ckpt Outputs/e2e_faster_rcnn_R-50-C4_1x/Jul30-15-51-27_node097_step/ckpt/model_step79999.pth \
#     --iter_size 2 \
#     --use_tfboard
