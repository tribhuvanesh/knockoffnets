# Knockoff Nets: Stealing Functionality of Black-Box Models, CVPR '19

**Tribhuvanesh Orekondy, Bernt Schiele Mario Fritz**

**Max Planck Institute for Informatics**

----

Machine Learning (ML) models are increasingly deployed in the wild to perform a wide range of tasks. 
In this work, we ask to what extent can an adversary steal functionality of such "victim" models based solely on blackbox interactions: image in, predictions out.
In contrast to prior work, we present an adversary lacking knowledge of train/test data used by the model, its internals, and semantics over model outputs.
We formulate model functionality stealing as a two-step approach: (i) querying a set of input images to the blackbox model to obtain predictions; and (ii) training a "knockoff" with queried image-prediction pairs.
We make multiple remarkable observations: (a) querying random images from a different distribution than that of the blackbox training data results in a well-performing knockoff; (b) this is possible even when the knockoff is represented using a different architecture; and (c) our reinforcement learning approach additionally improves query sample efficiency in certain settings and provides performance gains. 
We validate model functionality stealing on a range of datasets and tasks, as well as on a popular image analysis API where we create a reasonable knockoff for as little as $30.

**tl;dr**: We highlight a threat that functionality of blackbox models CNNs can easily 'knocked-off' under minimal assumptions
  

## Installation

### Environment
  * Python 3.6
  * Pytorch 1.1

Can be set up as:
```bash
$ conda env create -f environment.yml   # anaconda; or
$ pip install -r requirements.txt       # pip
```

### Datasets

You will need six datasets to perform all experiments in the paper, all extracted into the `data/` directory.
 * Victim datasets
   * Caltech256 ([Link](http://www.vision.caltech.edu/Image_Datasets/Caltech256/). Images in `data/256_ObjectCategories/<classname>/*.jpg`)
   * CUB-200-2011 ([Link](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html). Images in `data/CUB_200_2011/images/<classname>/*.jpg`)
   * Indoor Scenes ([Link](http://web.mit.edu/torralba/www/indoor.html). Images in `data/indoor/Images/<classname>/*.jpg`)
   * Diabetic Retinopathy ([Link](https://www.kaggle.com/c/diabetic-retinopathy-detection). Images in `data/diabetic_retinopathy/training_imgs/<classname>/*.jpg`)
 * Adversarial datasets
   * ImageNet ILSVRC 2012 ([Link](http://image-net.org/download-images). Images in `data/ILSVRC2012/training_imgs/<classname>/*.jpg`)
   * OpenImages ([Link](https://storage.googleapis.com/openimages/web/index.html). Images in `data/openimages/<classname>/*.jpg`)

## Attack: Overview

The commands/steps below will guide you to:
  1. Train victim models (or download pretrained models)
  1. Train knockoff models (or download pretrained models)
     1. Constructing transfer sets
     1. Training knockoff models 

## Victim Models
We follow the convention of storing victim models and related data (e.g., logs) under `models/victim/P_V-F_V/` (e.g., `cubs200-resnet34`).

### Option A: Download Pretrained Victim Models

Zip files (containing resnet-34 pytorch checkpoint `.pth.tar`, hyperparameters and training logs):
  * [Caltech256](https://datasets.d2.mpi-inf.mpg.de/orekondy19cvpr/victim_models/caltech256-resnet34.zip) (Accuracy = 78.4%)
  * [CUBS200](https://datasets.d2.mpi-inf.mpg.de/orekondy19cvpr/victim_models/cubs200-resnet34.zip)  (77.1%)
  * [Indoor67](https://datasets.d2.mpi-inf.mpg.de/orekondy19cvpr/victim_models/indoor67-resnet34.zip) (76.0%)
  * [Diabetic5](https://datasets.d2.mpi-inf.mpg.de/orekondy19cvpr/victim_models/diabetic5-resnet34.zip) (59.4%)

### Option B: Train Victim Models
 
```bash
# Format:
$ python knockoff/victim/train.py DS_NAME ARCH -d DEV_ID \
        -o models/victim/VIC_DIR -e EPOCHS --pretrained
# where DS_NAME = {cubs200, caltech256, ...}, ARCH = {resnet18, vgg16, densenet161, ...}
# if the machine contains multiple GPUs, DEV_ID specifies which GPU to use

# More details:
$ python knockoff/victim/train.py --help

# Example (CUB-200):
$ python knockoff/victim/train.py CUBS200 resnet34 -d 1 \
        -o models/victim/cubs200-resnet34 -e 10 --log-interval 25 \
        --pretrained imagenet
```

## Training Knockoff Models

We store the knockoff models and related data (e.g., transfer set, logs) under `data/adversary/P_V-F_A-pi/`  (e.g., `cubs200-resnet50-random`).

### Transfer Set Construction

```bash
# Format
$ python knockoff/adversary/transfer.py random models/victim/VIC_DIR \
        --out_dir models/adversary/ADV_DIR --budget BUDGET \
        --queryset QUERY_SET --batch_size 8 -d DEV_ID
# where QUERY_SET = {ImageNet1k ,...}

# More details
$ python knockoff/adversary/transfer.py --help

# Examples (CUB-200):
# Random
$ python knockoff/adversary/transfer.py random models/victim/cubs200-resnet34 \
        --out_dir models/adversary/cubs200-resnet34-random --budget 80000 \
        --queryset ImageNet1k --batch_size 8 -d 2
# Adaptive
$ python knockoff/adversary/transfer.py adaptive models/victim/cubs200-resnet34 \
        --out_dir models/adversary/cubs200-resnet34-random --budget 80000 \
        --queryset ImageNet1k --batch_size 8 -d 2
```

### Training Knock-offs

```bash
# Format:
$ python knockoff/adversary/train.py models/adversary/ADV_DIR ARCH DS_NAME \
        --budgets BUDGET1,BUDGET2,.. -d DEV_ID --pretrained --epochs EPOCHS \
        --lr LR
# DS_NAME refers to the dataset used to train victim model; used only to evaluate on test set during training of knockoff

# More details:
$ python knockoff/adversary/train.py --help

# Example (CUB-200)
$ python knockoff/adversary/train.py models/adversary/cubs200-resnet34-random \
        resnet34 CUBS200 --budgets 60000 -d 0 --pretrained imagenet \
        --log-interval 100 --epochs 200 --lr 0.01 
```

### Pretrained knock-off models

Zip files (containing pytorch checkpoint, transferset pickle file, hyperparameters and logs) can be downloaded using the links below.
Specifically, the knockoffs are resnet34s at B=60k using imagenet as the query set ($P_A$).

| $F_V$      | Random | Adaptive |
|------------|:--------:|:----------:|
| Caltech256 | [zip](https://datasets.d2.mpi-inf.mpg.de/orekondy19cvpr/adversary_models/caltech256-resnet34-imagenet-random-60k.zip) (76.0%)   | [zip](https://datasets.d2.mpi-inf.mpg.de/orekondy19cvpr/adversary_models/caltech256-resnet34-imagenet-adaptive-60k.zip) (%)    |
| CUBS200    | [zip](https://datasets.d2.mpi-inf.mpg.de/orekondy19cvpr/adversary_models/cubs200-resnet34-imagenet-random-60k.zip) (67.7%)   | [zip](https://datasets.d2.mpi-inf.mpg.de/orekondy19cvpr/adversary_models/cubs200-resnet34-imagenet-adaptive-60k.zip) (%)     |
| Indoor67   | [zip](https://datasets.d2.mpi-inf.mpg.de/orekondy19cvpr/adversary_models/indoor67-resnet34-imagenet-random-60k.zip) (68.2%)   | [zip](https://datasets.d2.mpi-inf.mpg.de/orekondy19cvpr/adversary_models/indoor67-resnet34-imagenet-adaptive-60k.zip) (%)     |
| Diabetic5  | [zip](https://datasets.d2.mpi-inf.mpg.de/orekondy19cvpr/adversary_models/diabetic5-resnet34-imagenet-random-60k.zip) (43.6%)   | [zip](https://datasets.d2.mpi-inf.mpg.de/orekondy19cvpr/adversary_models/diabetic5-resnet34-imagenet-adaptive-60k.zip) (%)     |


### Note
Since the current publicly available code uses an updated pytorch version and has been significantly refactored from the initially published version, expect minor differences in results.
Please contact me (see below) in case you want the exact pretrained models used in the paper. 


## Citation
If you found this work or code useful, please cite us:
```
@inproceedings{orekondy19knockoff,
    TITLE = {Knockoff Nets: Stealing Functionality of Black-Box Models},
    AUTHOR = {Orekondy, Tribhuvanesh and Schiele, Bernt and Fritz, Mario},
    YEAR = {2019},
    BOOKTITLE = {CVPR},
}
```


## Contact
In case of feedback, suggestions, or issues, please contact [Tribhuvanesh Orekondy](https://tribhuvanesh.github.io/)
