# INSTALL GUIDE
Clone the repo:

```
git clone git@github.com:AruniRC/detectron-self-train.git
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
