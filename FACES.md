# Face detection in CS6 videos

## Data preparation
CS6 has a total of 179 surveillance videos, taken from 9 different cameras. 
A symlink `data/CS6` should point to the CS6 data root location
(on Gypsum this is in `/mnt/nfs/scratch1/arunirc/data/CS6/CS6/CS6.0.01/CS6`). For the detection task, under `data/CS6` we need `protocols/cs6_face_detection_ground_truth.csv` that lists the ground-truth faces in videos and the folder `videos` that contains all 179 MP4 videos (29.6 GB).

Another symlink should point to the prepared annotations of CS6: `data/CS6_annot`, pointing to `/mnt/nfs/work1/elm/arunirc/Data/CS6_annots/` (on Gypsum). This contains extracted video frames (under `frames/<vid-name>/*.jpg`) and corresponding annotations in FDDB-style text files (as `video_annots/<vid-name>.txt`).

Video filenames for train(88)/val(5)/test(86) splits in `data/CS6/list_video_<split>.txt`.

**TODO**: script to convert from Janus-style formats in data/CS6 to FDDB-style ground-truth annotations in data/CS6_annots. 

* Validation experiments
* Mining
* Re-training
* Evaluation


## Validation experiments

These involve using 5 videos with full ground-truth annotations as validation to determine best hyper-parameter settings for the tracker and confidence thresholds for mining hard examples. All detections with CONF_SCORE > 0.5 are kept.

Output folder structure:
```
./Outputs/evaluations/frcnn-R-50-C4-1x/cs6/
    sample-baseline-video/
    mining-detections/
```


#### Detection and visualization

The script [tools/face/detect_video.py](tools/face/detect_video.py) detects faces in a video and optionally saves each frame with marked bounding-boxes as JPG images in a folder. By default the outputs are in `Outputs/evaluations/<model-name>/cs6/sample-baseline-video`, using *frcnn-R-50-C4-1x* as the model.

Example SLURM execution command: 
```
srun --pty --mem 100000 --gres gpu:1 python tools/face/detect_video.py \
--vis --video_name 801.mp4
```

This is run on the following videos: 501.mp4, 801.mp4, 1100.mp4, 3004.mp4, 3007.mp4. These are listed in `data/CS6/list_video_val.txt`.

Output folder structure:
```
./Outputs/evaluations/frcnn-R-50-C4-1x/cs6/
    sample-baseline-video/
        501/*.jpg
        801/*.jpg
        ...
        501.txt
        801.txt
        ...
```

**MP4 video from frames.** In the `sample-baseline-video` folder, use ffmpeg to convert the frames in a folder into a video (quicker than writing videos from Python). `ffmpeg -framerate 30 -pattern_type glob -i '501/*.jpg' -c:v libx24 501.mp4`.


#### Validating Tracklets

**Format conversion.** Each of the text files in `sample-baseline-video` needs to be converted into a "mining-format" compatible with the tracker code used for mining hard examples, using [tools/face/convert_dets_mining_format.py](tools/face/convert_dets_mining_format.py). The source code contains comments and usage examples. By default it uses videos in the "val" split of CS6 (as defined in `data/CS6/list_video_val.txt`). A range of thresholds are placed on the scores of the detections in `sample-baseline-video` folder. The resultant output structure is shown below:

```
./Outputs/evaluations/frcnn-R-50-C4-1x/cs6/
    sample-baseline-video/
    mining-detections/
        <split>_<conf-threshold>/
            501.txt
            801.txt
            ...
        ...
```

**Forming tracklets.**


**CS6 tracklet evaluation.** 

TODO 