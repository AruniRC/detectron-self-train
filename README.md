
# PyTorch Detectron for domain adaptation by self-training

This codebase replicates results for pedestrian detection with domain shifts on the BDD100k dataset, following the CVPR 2019 paper [Automatic adaptation of object detectors to new domains using self-training](http://vis-www.cs.umass.edu/unsupVideo/docs/self-train_cvpr2019.pdf). We provide trained models, train and eval scripts as well as splits of the dataset for download.

This repository is heavily based off [A Pytorch Implementation of Detectron](https://github.com/roytseng-tw/Detectron.pytorch). We modify it for experiments on domain adaptation face and pedestrian detectors. 


## Getting Started
Clone the repo:

```
git clone https://github.com/AruniRC/Detectron-pytorch-video.git
```

### Requirements

Tested under python3.

- python packages
  - pytorch>=0.3.1
  - torchvision>=0.2.0
  - cython
  - matplotlib
  - numpy
  - scipy
  - opencv
  - pyyaml
  - packaging
  - [pycocotools](https://github.com/cocodataset/cocoapi)  — for COCO dataset, also available from pip.
  - tensorboardX  — for logging the losses in Tensorboard
- An NVIDAI GPU and CUDA 8.0 or higher. Some operations only have gpu implementation.
- **NOTICE**: different versions of Pytorch package have different memory usages.


## Installation
This walkthrough describes setting up Detectron (3rd party pytorch implementation) repo. This setup assumes CUDA 8.0 and CuDNN 5.1, as well as the default C compiler being `gcc 5.4`. 

### Cluster environment
**Optional (please skip if not on a cluster)** If setting up on a SLURM cluster, please make sure that **only** these modules are loaded and not multiple versions of CUDA etc. that can cause build conflicts further on. List of loaded modules:

- slurm/16.05.8 
- gcc5/5.4.0
- cuda80/toolkit/8.0.61
- cudnn/5.1
- openmpi/gcc/64/1.10.1
- fftw2/openmpi/open64/64/float/2.1.5
- hdf5_18/1.8.17


### Create conda env

`conda create -n detectron-context python=3.5`

If you need to install conda, please follow [these instructions](https://docs.anaconda.com/anaconda/install/linux/).

Install pytorch and numpy using `pip`:
```
pip install https://download.pytorch.org/whl/cu80/torch-0.4.0-cp35-cp35m-linux_x86_64.whl
pip install numpy -I
```

### Test it out 
Start python at the command line and try to import torch (without errors):
```
$ python
>>> import torch
```

Rest of the packages:
```
pip install torchvision
pip install matplotlib
pip install scipy
pip install pyyaml
pip install cython
pip install pycocotools
pip install opencv-python
conda install cffi
pip install requests
pip install colorama

```

### Visualization installs

```
pip install tensorboardX
pip install tensorboard_logger
pip install tensorboard
```


### Compile Detectron-pytorch 
The makefile is in `lib/make.sh` w.r.t. the project root. Set the `CUDA_PATH`  to point to your local CUDA install (e.g `/usr/local/cuda`). If you want to use a CUDA library on different path, change that line accordingly.

It will compile all the modules you need, including NMS, ROI_Pooing, ROI_Crop and ROI_Align.

Note that, If you use `CUDA_VISIBLE_DEVICES` to set gpus, **make sure at least one gpu is visible when compiling the code.**

```
cd lib  # please change to this directory
sh make.sh
```

Make sure that there are no fatal errors in the output log of the make command above. Common issues are usually multiple versions of CUDA or CuDNN being present.

### Download Pretrained Backbone Model
Use ImageNet pretrained weights from Caffe for the backbone networks.
Download them and put them into the `{repo_root}/data/pretrained_model`, using the following command:

```
python tools/download_imagenet_weights.py
```

**NOTE**: Caffe pretrained weights have slightly better performance than Pytorch pretrained (the official Detectron also use pretrained weights from Caffe).



## Dataset
Create a data folder under the repo,

```
cd {repo_root}
mkdir data
```

### BDD-100k
Our pedestrian detection task uses both labeled and unlabeled data from the **Berkeley Deep Drive** [BDD-100k dataset](https://bdd-data.berkeley.edu/). Please register and download the dataset from their website. We use a symlink from our project root, `data/bdd100k` to link to the location of the downloaded dataset. The folder structure should be like this:

```
data/bdd100k/
    images/
        test/
        train/
        val/
    labels/
        train/
        val/
```

BDD-100k takes about 6.5 GB disk space. The 100k unlabeled videos take 234 GB space, but you do not need to download them, since we have already done the hard example mining on these and the extracted frames (+ pseudo-labels) are available for download.


### BDD Hard Examples
Mining the **hard positives** ("HPs") involve detecting pedestrians and tracklet formation on 100K videos. This was done on the UMass GPU Cluster and took about a week. We do not include this pipeline here (yet) -- the mined video frames and annotations are available for download as a gzipped tarball from [here](link_to_maxwell_tarball). **TODO** 

Now we create a symlink to the untarred BDD HPs from the project data folder, which should have the following structure: `data/bdd100k/*.jpg`. The image naming format is `<video-name>_<frame-number>.jpg`.



## Train and eval models

Use the environment variable `CUDA_VISIBLE_DEVICES` to control which GPUs to use. All the training scripts are run with 4 GPUs.

| Method  | Model weights |  Config YAML |  Train script |  Eval script | AP, AR |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Baseline | [bdd_baseline](http://maxwell.cs.umass.edu/self-train/models/bdd_ped_models/bdd_baseline/bdd_peds.pth)  | [cfg](configs/baselines/bdd100k.yaml)  |  [train](gypsum/scripts/train/bdd_scripts/bdd_baseline.sh)  |  [eval](gypsum/scripts/eval/bdd_scripts/baseline_source.sh)  |  15.21, xxx  |
| Dets | [bdd_dets](http://maxwell.cs.umass.edu/self-train/models/bdd_ped_models/bdd_dets/bdd_dets_model_step29999.pth)  | [cfg](configs/baselines/bdd_peds_dets_bs64_4gpu.yaml)  |  [train](gypsum/scripts/train/bdd_scripts/bdd_source_and_dets18k.sh)  |  [eval](gypsum/scripts/eval/bdd_scripts/bdd_dets_source.sh)  |  27.55, 56.90  |







### Adding a dataset

The general way to add a new dataset is to:
1. Add it to `lib/datasets/dataset_catalog.py`
2. Add it to `tools/train_net_step.py`

To get an idea using COCO 2017 as an example, search for `args.dataset == "coco2017":` in `tools/train_net_step.py` and for `coco_2017_train` in `tools/dataset_catalog.py`. The paths to the JSON training annotations are defined in `dataset_catalog.py`. 



### Verify by running on COCO-2017 
Put the Imagenet pre-trained models in `data/pretrained_model` by running `python tools/download_imagenet_weights.py`).

Then, verify setup by running COCO-2017 inference code:

```
CFG_PATH=configs/baselines/e2e_faster_rcnn_R-50-C4_1x.yaml
WT_PATH=/mnt/nfs/work1/elm/arunirc/Research/detectron-video/mask-rcnn.pytorch/data/detectron_trained_model/e2e_faster_rcnn_R-50-C4_1x.pkl

mkdir Outputs

srun --pty -p m40-long --gres gpu:4 --mem 100000 python tools/test_net.py \
--set TEST.SCORE_THRESH 0.1 TRAIN.JOINT_TRAINING False TRAIN.GT_SCORES False \
--multi-gpu-testing \
--dataset coco2017 \
--cfg ${CFG_PATH} \
--load_detectron ${WT_PATH} \
--output_dir Outputs
```






## Inference

### Visualize pre-trained Detectron model on images

This can run a pretrained Detectron model trained on MS-COCO categories, downloaded from the official Detectron Model Zoo, on the sample images. Note the `load_detectron` option to the `infer_simple.py` script, because we are using a Detectron model, not a checkpoint.

```
python tools/infer_simple.py --dataset coco --cfg cfgs/baselines/e2e_mask_rcnn_R-50-C4.yml --load_detectron {path/to/your/checkpoint} --image_dir {dir/of/input/images}  --output_dir {dir/to/save/visualizations}
```
`--output_dir` defaults to `infer_outputs`.



## Training


