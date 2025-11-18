#!/bin/bash

# accelerate the download speed (AutoDL)
source /etc/network_turbo

# download and unzip the dataset for aerial imagery segmentation
wget https://www.kaggle.com/api/v1/datasets/download/rupankarmajumdar/crop-pests-dataset

mv crop-pests-dataset crop-pests-dataset.zip

unzip crop-pests-dataset.zip

# install the required Python packages
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install scikit-learn matplotlib segmentation_models_pytorch


# 创建 train.sh 里面有 python train.py
# bash train.sh 2>&1 | tee err1.log