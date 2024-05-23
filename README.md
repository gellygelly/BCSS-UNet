# UNet For BCSS(Breast Cancer Semantic Segmentation)

### UNet Breast Cancer Image Segmentation 


[🩻BCSS dataset link] https://www.kaggle.com/datasets/whats2000breast-cancer-semantic-segmentation-bcss

This repo is based on the kaggle notebook code below.

[🤓reference code link] https://www.kaggle.com/code/whats2000/u-net-advance-bcss-segmentation-512

UNet: Convolutional Networks for Biomedical Image Segmentation

[Paper] https://arxiv.org/abs/1505.04597

## 1. Data Preparation

### Directory Structure 
```bash
data
├── BCSS-UNet
│   ├── train
│   │   ├──1.png
│   │   ├──aaa.png
│   ├── train_mask
│   │   ├──1.png
│   │   ├──aaa.png
│   ├── val
│   ├── val_mask
│   ├── test
│   ├── test_mask
```

You can revise the train, val, test dataset folder path in config.py 


### CHECK!
The file name of the mask image file is the same as the image file name.
Mask image file consists of 0-255 values, and each pixel is shown gt label. 
If you want to use other dataset(except for bcss dataset), you should use this format.

## 2. Get started

### How to use this code

1. clone this repo and install packages
```python
# git clone
git clone [This Repository URL]

# package install
pip install -r requirements.txt
```
The code is stable using Pytorch 2.1.0, cuda 11.8.0

[🐳dockerhub link] https://hub.docker.com/layers/pytorch/pytorch/2.1.0-cuda11.8-cudnn8-devel/images/sha256:558b78b9a624969d54af2f13bf03fbad27907dbb6f09973ef4415d6ea24c80d9

2. set config file

you can set DATASET PATH and hyperparameters in config.py

3. train & test 

```python
python train.py
python test.py
```

4. check model result in log folder

You can check the train/val loss/mIoU log and graph image in the 'log/train' folder.
You can also view model test results (including metrics performance) and pred mask images in the 'log/test' folder.
