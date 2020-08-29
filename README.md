<!-- PROJECT LOGO -->
<p align="center">
  <h3 align="center">Deep Edge Aware Filters (DEAF)</h3>

  <p align="center">
    PyTorch implementation
    <br />
    <br />
    <a href="https://github.com/nrupatunga/pytorch-deaf/issues">Report Bug</a>
    ·
    <a href="https://github.com/nrupatunga/pytorch-deaf/issues">Request Feature</a>
  </p>
</p>

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [App Demo](#app-demo)
* [Getting Started](#getting-started)
	- [Code Setup](#code-setup)
	- [Data Download](#data-download)
	- [Training](#training)
	- [Testing](#testing)


<!-- ABOUT THE PROJECT -->
## About The Project

This repository is an attempt to implement ["Deep Edge Aware
Filters"](http://proceedings.mlr.press/v37/xub15.pdf) in PyTorch.

The official Matlab implementation is in this
[link](https://github.com/jimmy-ren/vcnn_double-bladed/tree/master/applications/deep_edge_aware_filters)

<!-- APP DEMO-->
#### App Demo
| DEAF |
|------------------------|
|![](https://github.com/nrupatunga/pytorch-deaf/blob/master/scripts/demo/deaf.gif)|


<!-- GETTING STARTED -->
## Getting Started

#### Code Setup
```
# Clone the repository
$ git clone https://github.com/nrupatunga/pytorch-deaf

# install all the required repositories
$ cd pytorch-deaf
$ pip install -r requirements.txt

# Add current directory to environment
$ source settings.sh
```

#### Data Download

Since the author didn't release original data trained with, I used
[DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) to prepare
data for this project. Once you download the data, split into `train`
and `val` images. In my case I split first `800` for training and
remaining `100` for validation.

To prepare data run:

```
$ cd pytorch-deaf/src/utility/

# change the `root_dir` and `save_dir` in `gen_training_data.py` and
# then run

$ python gen_training_data.py
```

#### Training
```
$ cd pytorch-deaf/scripts/exp-1

# Modify the DATA_PATH variable in the script in train.sh
# run to train
$ bash train.sh
```
**Note**: I trained for 8 epochs, it will take half a day to train, please
have your data on SSD for better training performance

#### Testing
```
$ cd pytorch-deaf/scripts/exp-1

# Change the model path `ckpt_path` in app.py and

$ python app.py
```
