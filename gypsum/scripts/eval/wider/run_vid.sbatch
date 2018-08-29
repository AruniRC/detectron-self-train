#!/bin/bash
#
#SBATCH --job-name=wider_video
#SBATCH -o gypsum/logs/%j_wider_video.out            # File to which STDOUT will be written
#SBATCH -e gypsum/errs/%j_wider_video.err            # File to which STDERR will be written
#SBATCH -p m40-long
#SBATCH --gres=gpu:1
#SBATCH --mem=80000
#SBATCH --exclude=node025


## Usage (from project root):


echo 'genre: '$1', vid-folder: '$2', video: '$3 

time ./tools/face/detect_video_wider.py \
    --genre $1 \
    --data_dir $2 \
    --video_name $3 \
    --model_name $4 \
    --cfg $5 \
    --load_ckpt $6

    





