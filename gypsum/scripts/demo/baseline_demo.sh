#!/usr/bin/env bash
#SBATCH --job-name=basleine_source_eval
#SBATCH -o gypsum/logs/%j_baseline_source_eval.txt 
#SBATCH -e gypsum/errs/%j_baseline_source_eval.txt
#SBATCH -p 1080ti-short
#SBATCH --gres=gpu:1
#SBATCH --mem=100000
##SBATCH --cpus-per-task=4
##SBATCH --mem-per-cpu=4096


MODEL_PATH=/mnt/nfs/scratch1/pchakrabarty/bdd_recs/ped_models/bdd_peds.pth


python  tools/infer_demo.py \
    --model_name baseline \
    --cfg configs/baselines/bdd100k.yaml \
    --load_ckpt ${MODEL_PATH} \
    --image_dir demo/BDD/images
