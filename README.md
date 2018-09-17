# A Pytorch Implementation of Detectron

[![Build Status](https://travis-ci.com/roytseng-tw/Detectron.pytorch.svg?branch=master)](https://travis-ci.com/roytseng-tw/Detectron.pytorch)


**This code follows the implementation architecture of Detectron.** Only part of the functionality is supported. Check [this section](#supported-network-modules) for more information.


## News

- (2018/05/25) Support ResNeXt backbones.
- (2018/05/22) Add group normalization baselines.
- (2018/05/15) PyTorch0.4 is supported now !

## Getting Started
Clone the repo:

```
git clone https://github.com/roytseng-tw/mask-rcnn.pytorch.git
```
BRanch test

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

### Setup on Gypsum
I used Anaconda to get all these requirements set up.

```
$ module list
Currently Loaded Modulefiles:
  1) slurm/16.05.8                         5) cuda80/toolkit/8.0.61
  2) openmpi/gcc/64/1.10.1                 6) cudnn/5.1
  3) hdf5/1.6.10                           7) openblas/dynamic/0.2.18
  4) fftw2/openmpi/open64/64/float/2.1.5   8) hdf5_18/1.8.17
```

The total list of items in the conda environment is there in `conda_list.txt` in the project root folder.



### Compilation

Compile the CUDA code (the makefile is modified to work on the UMass Gypsum cluster environment):

```
cd lib  # please change to this directory
sh make.sh
```

`CUDA_PATH` defaults to `/usr/loca/cuda`. If you want to use a CUDA library on different path, change this [line](https://github.com/roytseng-tw/mask-rcnn.pytorch/tree/master/lib/make.sh#L3) accordingly.

It will compile all the modules you need, including NMS, ROI_Pooing, ROI_Crop and ROI_Align. (Actually gpu nms is never used ...)

Note that, If you use `CUDA_VISIBLE_DEVICES` to set gpus, **make sure at least one gpu is visible when compile the code.**

### Data Preparation

Create a data folder under the repo,

```
cd {repo_root}
mkdir data
```

Create symlink at `data/WIDER` to `/mnt/nfs/scratch1/arunirc/data/WIDER`.

```
data/WIDER
    WIDER_val/
    WIDER_train/
    wider_face_split/
    wider_face_train_imlist.txt
    wider_face_val_imlist.txt
    wider_face_train_annot.txt
    wider_face_train_annot_coco_style.json
```

If missing, the last JSON file needs to be created *once* by running `python ./lib/datasets/wider/convert_face_to_coco.py --dataset wider --outdir data/WIDER --datadir data/WIDER`. This converts the FDDB-style face bounding-box annotations of `wider_face_train_annot.txt` into the MS-COCO style JSON needed for training Detectron models.


### Pretrained Model

Use ImageNet pretrained weights from Caffe for the backbone networks.

- [ResNet50](https://drive.google.com/open?id=1wHSvusQ1CiEMc5Nx5R8adqoHQjIDWXl1), [ResNet101](https://drive.google.com/open?id=1x2fTMqLrn63EMW0VuK4GEa2eQKzvJ_7l), [ResNet152](https://drive.google.com/open?id=1NSCycOb7pU0KzluH326zmyMFUU55JslF)
- [VGG16](https://drive.google.com/open?id=19UphT53C0Ua9JAtICnw84PPTa3sZZ_9k)  (vgg backbone is not implemented yet)

Download them and put them into the `{repo_root}/data/pretrained_model`.

You can the following command to download them all:

- extra required packages: `argparse_color_formater`, `colorama`, `requests`

```
python tools/download_imagenet_weights.py
```

**NOTE**: Caffe pretrained weights have slightly better performance than Pytorch pretrained. Suggest to use Caffe pretrained models from the above link to reproduce the results. By the way, Detectron also use pretrained weights from Caffe.

**If you want to use pytorch pre-trained models, please remember to transpose images from BGR to RGB, and also use the same data preprocessing (minus mean and normalize) as used in Pytorch pretrained model.**



## Inference

### Visualize pre-trained Detectron model on images

This can run a pretrained Detectron model trained on MS-COCO categories, downloaded from the official Detectron Model Zoo, on the sample images. Note the `load_detectron` option to the `infer_simple.py` script, because we are using a Detectron model, not a checkpoint.

```
python tools/infer_simple.py --dataset coco --cfg cfgs/baselines/e2e_mask_rcnn_R-50-C4.yml --load_detectron {path/to/your/checkpoint} --image_dir {dir/of/input/images}  --output_dir {dir/to/save/visualizations}
```
`--output_dir` defaults to `infer_outputs`.


### Evaluate the trained model checkpoint on WIDER-val set

Example evaluation script
```
WT_PATH=Outputs/e2e_faster_rcnn_R-50-C4_1x/Jul30-15-51-27_node097_step/ckpt/model_step79999.pth
CFG_PATH=configs/wider_face/e2e_faster_rcnn_R-50-C4_1x.yaml

python tools/eval/run_face_detection_on_wider.py \
  --cfg ${CFG_PATH} \
  --load_ckpt ${WT_PATH} \
  --exp_name frcnn-R-50-C4-1x
```




## Training

Use the environment variable `CUDA_VISIBLE_DEVICES` to control which GPUs to use.

### Adapative config adjustment

#### Let's define some terms first

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; batch_size:            `NUM_GPUS` x `TRAIN.IMS_PER_BATCH`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; effective_batch_size:  batch_size x `iter_size`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; change of somethining: `new value of something / old value of something`

Following config options will be adjusted **automatically** according to actual training setups: 1) number of GPUs `NUM_GPUS`, 2) batch size per GPU `TRAIN.IMS_PER_BATCH`, 3) update period `iter_size`

- `SOLVER.BASE_LR`: adjust directly propotional to the change of batch_size.
- `SOLVER.STEPS`, `SOLVER.MAX_ITER`: adjust inversely propotional to the change of effective_batch_size.


### Train from scratch
Take mask-rcnn with res50 backbone for example.
```
python tools/train_net_step.py --dataset coco2017 --cfg configs/baselines/e2e_mask_rcnn_R-50-C4.yml --use_tfboard --bs {batch_size} --nw {num_workers}
```

Use `--bs` to overwrite the default batch size to a proper value that fits into your GPUs. Simliar for `--nw`, number of data loader threads defaults to 4 in config.py.

Specify `—-use_tfboard` to log the losses on Tensorboard.

Training for **WIDER-Face**.
```
python tools/train_net_step.py \
    --dataset wider_train \
    --cfg configs/wider_face/e2e_faster_rcnn_R-50-C4_1x.yaml  \
    --use_tfboard
```



### The use of `--iter_size`
As in Caffe, update network once (`optimizer.step()`) every `iter_size` iterations (forward + backward). This way to have a larger effective batch size for training. Notice that, step count is only increased after network update.

```
python tools/train_net_step.py --dataset coco2017 --cfg configs/baselines/e2e_mask_rcnn_R-50-C4.yml --bs 4 --iter_size 4
```
`iter_size` defaults to 1.

### Finetune from a pretrained checkpoint
```
python tools/train_net_step.py ... --load_ckpt {path/to/the/checkpoint}
```
or using Detectron's checkpoint file
```
python tools/train_net_step.py ... --load_detectron {path/to/the/checkpoint}
```

### Resume training with the same dataset and batch size
```
python tools/train_net_step.py ... --load_ckpt {path/to/the/checkpoint} --resume
```
When resume the training, **step count** and **optimizer state** will also be restored from the checkpoint. For SGD optimizer, optimizer state contains the momentum for each trainable parameter.

**NOTE**: `--resume` is not yet supported for `--load_detectron`

### Set config options in command line
```
  python tools/train_net_step.py ... --no_save --set {config.name1} {value1} {config.name2} {value2} ...
```
- For Example, run for debugging.
  ```
  python tools/train_net_step.py ... --no_save --set DEBUG True
  ```
  Load less annotations to accelarate training progress. Add `--no_save` to avoid saving any checkpoint or logging.

### Show command line help messages
```
python train_net_step.py --help
```

### Two Training Scripts

In short, use `train_net_step.py`.

In `train_net_step.py`:
- `SOLVER.LR_POLICY: steps_with_decay` is supported.
- Training warm up in [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677) is supported.

(Deprecated) In `train_net.py` some config options have no effects and worth noticing:

 - `SOLVER.LR_POLICY`, `SOLVER.MAX_ITER`, `SOLVER.STEPS`,`SOLVER.LRS`:
  For now, the training policy is controlled by these command line arguments:

    - **`--epochs`**: How many epochs to train. One epoch means one travel through the whole training sets. Defaults to  6.
    - **`--lr_decay_epochs `**: Epochs to decay the learning rate on. Decay happens on the beginning of a epoch. Epoch is 0-indexed. Defaults to [4, 5].

   For more command line arguments, please refer to `python train_net.py --help`

- `SOLVER.WARM_UP_ITERS`, `SOLVER.WARM_UP_FACTOR`, `SOLVER.WARM_UP_METHOD`:
  Training warm up is not supported.


## Supported Network modules

- Backbone:
  - ResNet:
    `ResNet50_conv4_body`,`ResNet50_conv5_body`,
    `ResNet101_Conv4_Body`,`ResNet101_Conv5_Body`,
    `ResNet152_Conv5_Body`
  - ResNeXt:
    `[fpn_]ResNet101_Conv4_Body`,`[fpn_]ResNet101_Conv5_Body`, `[fpn_]ResNet152_Conv5_Body`
  - FPN:
    `fpn_ResNet50_conv5_body`,`fpn_ResNet50_conv5_P2only_body`,
    `fpn_ResNet101_conv5_body`,`fpn_ResNet101_conv5_P2only_body`,`fpn_ResNet152_conv5_body`,`fpn_ResNet152_conv5_P2only_body`

- Box head:
  `ResNet_roi_conv5_head`,`roi_2mlp_head`, `roi_Xconv1fc_head`, `roi_Xconv1fc_gn_head`

- Mask head:
  `mask_rcnn_fcn_head_v0upshare`,`mask_rcnn_fcn_head_v0up`, `mask_rcnn_fcn_head_v1up`, `mask_rcnn_fcn_head_v1up4convs`, `mask_rcnn_fcn_head_v1up4convs_gn`

- Keypoints head:
  `roi_pose_head_v1convX`

**NOTE**: the naming is similar to the one used in Detectron. Just remove any prepending `add_`.


## Configuration Options

Architecture specific configuration files are put under [configs](configs/). The general configuration file [lib/core/config.py](lib/core/config.py) **has almost all the options with same default values as in Detectron's**, so it's effortless to transform the architecture specific configs from Detectron.

**Some options from Detectron are not used** because the corresponding functionalities are not implemented yet. For example, data augmentation on testing.

### Extra options
- `MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = True`:  Whether to load ImageNet pretrained weights.
  - `RESNETS.IMAGENET_PRETRAINED_WEIGHTS = ''`: Path to pretrained residual network weights. If start with `'/'`, then it is treated as a absolute path. Otherwise, treat as a relative path to `ROOT_DIR`.
- `TRAIN.ASPECT_CROPPING = False`, `TRAIN.ASPECT_HI = 2`, `TRAIN.ASPECT_LO = 0.5`: Options for aspect cropping to restrict image aspect ratio range.
- `RPN.OUT_DIM_AS_IN_DIM = True`, `RPN.OUT_DIM = 512`, `RPN.CLS_ACTIVATION = 'sigmoid'`: Official implement of RPN has same input and output feature channels and use sigmoid as the activation function for fg/bg class prediction. In [jwyang's implementation](https://github.com/jwyang/faster-rcnn.pytorch/blob/master/lib/model/rpn/rpn.py#L28), it fix output channel number to 512 and use softmax as activation function.

### How to transform configuration files from Detectron

1. Remove `MODEL.NUM_CLASSES`. It will be set according to the dataset specified by `--dataset`.
2. Remove `TRAIN.WEIGHTS`, `TRAIN.DATASETS` and `TEST.DATASETS`
3. For module type options (e.g `MODEL.CONV_BODY`, `FAST_RCNN.ROI_BOX_HEAD` ...), remove `add_` in the string if exists.
4. If want to load ImageNet pretrained weights for the model, add `RESNETS.IMAGENET_PRETRAINED_WEIGHTS` pointing to the pretrained weight file. If not, set `MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS` to `False`.
5. [Optional] Delete `OUTPUT_DIR: .` at the last line
6. Do **NOT** change the option `NUM_GPUS` in the config file. It's used to infer the original batch size for training, and learning rate will be linearly scaled according to batch size change. Proper learning rate adjustment is important for training with different batch size.
7. For group normalization baselines, add `RESNETS.USE_GN: True`.

## My nn.DataParallel

- **Keep certain keyword inputs on cpu**
  Official DataParallel will broadcast all the input Variables to GPUs. However, many rpn related computations are done in CPU, and it's unnecessary to put those related inputs on GPUs.
- **Allow Different blob size for different GPU**
  To save gpu memory, images are padded seperately for each gpu.
- **Work with returned value of dictionary type**

## Benchmark
[BENCHMARK.md](BENCHMARK.md)
