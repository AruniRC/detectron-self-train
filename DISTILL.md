Branch to try out Knowledge Distillation loss with Detectron


## Distillation Annotations with Score


The detections (with scores) are saved under `Output/evaluations/.../<method-name>/annot-format-dets/cs6_annot_train_scores.txt`. This file contains all the bounding-boxes and their detection scores.

Use the `lib/datasets/wider/convert_face_to_coco.py` script to take CS6 detections outputs and convert to a network-training JSON that has a "scores" field along with bounding-box annotations. The list of existing accepted datasets (and their scores counterparts) are in the script file.