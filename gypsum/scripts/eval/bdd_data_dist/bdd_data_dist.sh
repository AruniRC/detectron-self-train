#!/usr/bin/env bash
#SBATCH --job-name=eval_bdd_data_dist
#SBATCH -o gypsum/logs/%j_eval_bdd_data_dist.txt 
#SBATCH -e gypsum/errs/%j_eval_bdd_data_dist.txt
#SBATCH -p 1080ti-long
#SBATCH --gres=gpu:8
#SBATCH --mem=100000
##SBATCH --cpus-per-task=4
##SBATCH --mem-per-cpu=4096

ROOT=/mnt/nfs/scratch1/pchakrabarty/bdd_recs/rebuttal_results/data_distill/bdd_data_dist
DATASET_NAME=bdd_data_dist

python tools/test_net.py \
    --dataset bdd_peds_dets18k_target_domain \
    --cfg configs/baselines/bdd_data_dist.yaml \
    --output_dir $ROOT/results \
    --set TRAIN.JOINT_TRAINING False TRAIN.GT_SCORES False \
    --multi-gpu-testing \
    --load_ckpt /mnt/nfs/scratch1/pchakrabarty/ped_models/bdd_peds.pth \

# make a json with the predictions
python tools/res2json.py \
    --results_dir $ROOT/results \
    --eval_json_file data/bdd_jsons/bdd_dets18k.json \
    --output_dir $ROOT \
    --dataset_name $DATASET_NAME \
# rename json file
mv $ROOT/bbox_bdd_peds_dets18k_target_domain_results.json $ROOT/$DATASET_NAME.json
                    
# visualize the created json
python lib/datasets/bdd100k/viz_json.py \
    --output_dir $ROOT/viz \
    --json_file $ROOT/$DATASET_NAME.json \
    --imdir /mnt/nfs/scratch1/pchakrabarty/bdd_HP18k \


