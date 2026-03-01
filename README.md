# HRAD

This repository is the offical implementation for "HRAD: Towards Region-Adaptive Disentanglement
for Referring Remote Sensing Image Segmentation" 

## contributions

We introduce RefOPT, a new benchmark tailored for language-conditioned grounding and pixel-level delineation under challenging remote-sensing observations, and propose HRAD—a unified framework integrating Global Prior Generation (GPG), Progressive Feature Recursion (PFR), and hierarchical Region-Adaptive Feature Disentanglement (RFD)—to mitigate scale variation, regional heterogeneity, and partial observability, achieving consistent state-of-the-art performance across multiple datasets.

## RefOPT

RefOPT is a remote-sensing dataset for language-guided object localization/segmentation. Your local copy includes images, bounding boxes, English descriptions, and instance segmentation. In the folder, images/ stores the raw images, annotations/ contains XML annotations aligned with images, annotations.txt/split/*.txt provide id + bbox + text description, mask_isat/ provides polygon segmentations, mask/ provides mask images, and instances.json records metadata such as image sizes.

## Setting Up
### Preliminaries
The code has been verified to work with PyTorch v1.12.1 and Python 3.7.
1. Clone this repository.
2. Change directory to root of this repository.
### Package Dependencies
1. Create a new Conda environment with Python 3.7 then activate it:
```shell
conda create -n FIANet python==3.7
conda activate FIANet
```

2. Install PyTorch v1.12.1 with a CUDA version that works on your cluster/machine (CUDA 10.2 is used in this example):
```shell
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
```

3. Install the packages in `requirements.txt` via `pip`:
```shell
pip install -r requirements.txt
```
### The Initialization Weights for Training
1. Create the `./pretrained_weights` directory where we will be storing the weights.
```shell
mkdir ./pretrained_weights
```
2. Download [pre-trained classification weights of
the Swin Transformer](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth),
and put the `pth` file in `./pretrained_weights`.
These weights are needed for training to initialize the visual encoder.
3. Download [BERT weights from HuggingFace’s Transformer library](https://huggingface.co/google-bert/bert-base-uncased), 
and put it in the root directory. 

## Datasets
We perform the experiments on two dataset including [RefSegRS](https://github.com/zhu-xlab/rrsis) and [RRSIS-D](https://github.com/Lsan2401/RMSIN). 

## Training
We use one GPU to train our model. 
For training on RefSegRS dataset:
```shell
python train.py --dataset refsegrs --model_id FIANet --epochs 60 --lr 5e-5 --num_tmem 1  
```

For training on RRSIS-D dataset:
```shell
python train.py --dataset rrsisd --model_id FIANet --epochs 40 --lr 3e-5 --num_tmem 3  
```
The pretrained models can be downloaded from [[BaiduNetDisk](https://pan.baidu.com/s/1WgvKFn9nXiny1pzcvVJjwQ?pwd=65g4)](extract code: 65g4).

## Testing
For RefSegRS dataset:
```shell
python test.py --swin_type base --dataset refsegrs --resume ./your_checkpoints_path --split test --window12 --img_size 480 --num_tmem 1 
```
For RRSIS-D dataset:
```shell
python test.py --swin_type base --dataset rrsisd --resume ./your_checkpoints_path --split test --window12 --img_size 480 --num_tmem 3
```
