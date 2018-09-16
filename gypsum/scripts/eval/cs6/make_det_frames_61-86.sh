#!/usr/bin/env bash



###    Save frames with detections for all videos listed in "train" split of CS6

for VIDEO in `cat data/CS6/list_video_train.txt| sed -n "61,86p"`
do
    echo $VIDEO

    srun --mem 20000 -p titanx-long -o %j_det_frame.txt -e %j_det_frame.txt \
	python tools/face/make_det_frames.py --video_name $VIDEO &

    # sbatch gypsum/scripts/eval/cs6/run_det_video.sbatch \
    #     $VIDEO \
    #     ${DET_NAME} \
    #     ${OUT_DIR} \
    #     ${CONF_THRESH}

done
