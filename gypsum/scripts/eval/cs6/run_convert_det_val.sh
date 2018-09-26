#!/usr/bin/env bash


## --- CS6 + WIDER ---


DET_NAME="train-CS6-train-HP+WIDER-bs512-gpu4-10k_val-easy_conf-0.1"
DET_DIR="Outputs/evaluations/frcnn-R-50-C4-1x/cs6/"${DET_NAME}
SPLIT="val_easy"

srun --mem 50000 python tools/face/make_cs6_det_eval.py \
    --split $SPLIT \
    --det_dir ${DET_DIR}
