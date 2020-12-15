# Code Documentation

## Installation
Clone this repository into any place you want.
```bash
git clone https://github.com/authierj/EDSR-PyTorch
cd EDSR-PyTorch
```
**Dependencies**
* Python 3.8.5
* PyTorch >= 1.0.0
* numpy
* skimage
* imageio
* matplotlib
* tqdm
* cv2 >= 3.xx (Only if you want to use video input/output)

## Datasets
We used the [DIV2K] dataset (https://data.vision.ee.ethz.ch/cvl/DIV2K/) to train our models and the wildely used benchmark datasets [Set5](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html), [Set14](), [B100](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) and [Urban100]() to test the models. (still need to add the links)

## Pre-trained models
The models used for the experiments can be found in [experiment](/experiment) under <name_model>/model_best.pt

## Demo
You can test a SR resolution alogrithms with your images. Place your image in the ```test```folder. The network support .png and .jpeg files
Run the following line in ```src```

```bash
python main.py --data_test Demo --scale 4 --pre_train ../experiment/task4x2/model/model_best.pt --test_only --save_results
```
You can find the results in ```experiment/test```

If you want to train a network on the DIV2K dataset and test it on the Urban100 benchmark run the follwing lines in your terminal
```bash
cd src       # You are now in */EDSR-PyTorch/src
python main.py --model <MODEL_NAME> --scale 2 --n_resblock 8 --save <...> --save_results --reset
```
If you wish it you can change the scale, 3 and 4 are mainly used, as well as the n_resblock to have deeper network, many other options are possible and are listed in ```option.py```

To test the performance of a pre-trained network run the following lines in your terminal
```bash
cd src       # You are now in */EDSR-PyTorch/src
python main.py --test_only --pre_train ../experiment/<name_network>/model/model_best.pt --data_test <benchmark dataset> --save <...> --save_results --reset
```


# Project Documentation

## Introduction

Due to the rise of mobile devices like drones and smartphones camera sensors must be made smaller and more lightweight, this leads to a decrease in image resolution and quality. This where single image super resolution which aims at recovering a high resolution (HR) images from the degraded low resolution (LR) images comes in handy. In this work we look at the performance of super resolution convolutional networks (SRCNN) which are trained on a large dataset of pairwised LR and HR images and aims to minimize a loss function that gardes the performance of the super resolution (SR) images, the outputs of the network, compared to the HR images and then tested on different datasets. The resulting networks can then be used in mobile devices to increase the quality of the LR images.


Alternative:(On the other hand, also due to limited storage limitations, in remote systems, or due to limited data transfer, images are being saved in a lower resolution format. But for further tasks on bigger computers which don’t have these issues we again want to work with higher resolution. This is where super resolution comes in handy. In contrary to many other processes image super resolution creates new data. That means it creates more outputs than it has inputs. In the downsampling process data is lost and the Super resolution network can’t make use of this data.)

## Baseline

Our project is based on the [EDSR-PyTorrch](https://github.com/thstkdgus35/EDSR-PyTorch) repository which allready includes some Loss-functions, basic blocks, dataloaders and training and testing functions. In addition we use a [downsampling function](https://github.com/ofsoundof/dhp/blob/master/restoration/data/div2ksub.py) from an other repositry. We added some new simpler models to the repositry and embeded the downsampling function into the repositry with some minor changes.

## Background

There are two general aproaches for solving the super resolution problem. First you can interpolate your low resolution image using for example bicubic interpolation, and then run a convolutional neural network (CNN) on the interpolated images. Or you can add an upsampling block at the end of the CNN, which makes the network smaller and therefor faster. The EDSR model, which we also tested to see how the dataloaoding, training, testing, etc. works, uses the second approach. In this project both approches are tested, while using the first approach only for the simpler networks. 

## Method & experiment

For the training of the networks we use HR images, which we degrade with a simplified model including blurring, downsampling with bicubic interpolation and noise, to get paired training data (LR, HR). Then we train our network on a large datasets of images to try do the inverse process and reconstituing a HR image. In fact the network is trained to exctract high frequencies information from a low-frequency input.

**Networks with a bicubic interpolated input**

![](/figs/basic_blocks.png)

The firsts SRCNN networks we implemented were three layers CNN with blocks consisiting of a convolution followed by a ReLU activation function and the input of the network is a LR image interpolated with a bicubic interpolation to have to size of the HR image. 



results in a table 

| Model | Scale | File name (.pt) | number of layers | SET5 PSNR | Set14 PSNR | Urban100 PSNR | B100 PSNR |
|  ---  |  ---  | ---       | ---        | ---  | ---        | ---  | ---        |
| **Bicubic** | 2 | bicubic_input_model_x2 | 3 | 34.61 dB |
| | 3 | bicubic_input_model_x3 | 3 | 30.92 dB |
| | 4 | bicubic_input_model_x4 | 3 | 28.95 dB |
| | 2 | bicubic_input_model_x2 | 8 | 35.03 dB |
| | 2 | bicubic_input_model_grc_x2 | 8 | 31.26 dB |
| **SRCNN** | 2 | srcnn_basic | 3.23 M | 34.63 dB |
| | 2 | srcnn_basic | 7.95 M| 34.92 dB |
| | 2 | srcnn_basic | | 30.94 dB |
| | 2 | srcnn_basic | | 31.22 dB |
| | 2 | srcnn_basic | | 28.97 dB |
| | 2 | srcnn_basic | | 29.24 dB |
| | 2 | task4 | 8 | 26.896 dB | 25.964 dB | 24.243 dB | 31.941 dB |
| | 2 | task6 | 8 | results strange... dB | ... dB | ... dB | 32.014 dB |
| | 2 | leakyReLU| 8 | 28.151 dB | 26.931 dB | 25.078 dB | 31.945 dB |
| | 2 | PreLU | 8 | 27.652 dB | 26.728 dB | 24.922 dB | 31.944 dB |
| | 2 | task10 | 8 | 30.119 dB | 28.830 dB | 26.311 dB | 32.215 dB |




original image and SR images of models

## Conclusion


**About PyTorch 1.2.0**
  * Now the master branch supports PyTorch 1.2.0 by default.
  * Due to the serious version problem (especially torch.utils.data.dataloader), MDSR functions are temporarily disabled. If you have to train/evaluate the MDSR model, please use legacy branches.

# EDSR-PyTorch

**About PyTorch 1.1.0**
  * There have been minor changes with the 1.1.0 update. Now we support PyTorch 1.1.0 by default, and please use the legacy branch if you prefer older version.

![](/figs/main.png)

This repository is an official PyTorch implementation of the paper **"Enhanced Deep Residual Networks for Single Image Super-Resolution"** from **CVPRW 2017, 2nd NTIRE**.
You can find the original code and more information from [here](https://github.com/LimBee/NTIRE2017).

If you find our work useful in your research or publication, please cite our work:

[1] Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee, **"Enhanced Deep Residual Networks for Single Image Super-Resolution,"** <i>2nd NTIRE: New Trends in Image Restoration and Enhancement workshop and challenge on image super-resolution in conjunction with **CVPR 2017**. </i> [[PDF](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Lim_Enhanced_Deep_Residual_CVPR_2017_paper.pdf)] [[arXiv](https://arxiv.org/abs/1707.02921)] [[Slide](https://cv.snu.ac.kr/research/EDSR/Presentation_v3(release).pptx)]
```
@InProceedings{Lim_2017_CVPR_Workshops,
  author = {Lim, Bee and Son, Sanghyun and Kim, Heewon and Nah, Seungjun and Lee, Kyoung Mu},
  title = {Enhanced Deep Residual Networks for Single Image Super-Resolution},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {July},
  year = {2017}
}
```
We provide scripts for reproducing all the results from our paper. You can train your model from scratch, or use a pre-trained model to enlarge your images.

**Differences between Torch version**
* Codes are much more compact. (Removed all unnecessary parts.)
* Models are smaller. (About half.)
* Slightly better performances.
* Training and evaluation requires less memory.
* Python-based.

## Dependencies
* Python 3.6
* PyTorch >= 1.0.0
* numpy
* skimage
* **imageio**
* matplotlib
* tqdm
* cv2 >= 3.xx (Only if you want to use video input/output)

## Code
Clone this repository into any place you want.
```bash
git clone https://github.com/thstkdgus35/EDSR-PyTorch
cd EDSR-PyTorch
```

## Quickstart (Demo)
You can test our super-resolution algorithm with your images. Place your images in ``test`` folder. (like ``test/<your_image>``) We support **png** and **jpeg** files.

Run the script in ``src`` folder. Before you run the demo, please uncomment the appropriate line in ```demo.sh``` that you want to execute.
```bash
cd src       # You are now in */EDSR-PyTorch/src
sh demo.sh
```

You can find the result images from ```experiment/test/results``` folder.

| Model | Scale | File name (.pt) | Parameters | ****PSNR** |
|  ---  |  ---  | ---       | ---        | ---  |
| **EDSR** | 2 | EDSR_baseline_x2 | 1.37 M | 34.61 dB |
| | | *EDSR_x2 | 40.7 M | 35.03 dB |
| | 3 | EDSR_baseline_x3 | 1.55 M | 30.92 dB |
| | | *EDSR_x3 | 43.7 M | 31.26 dB |
| | 4 | EDSR_baseline_x4 | 1.52 M | 28.95 dB |
| | | *EDSR_x4 | 43.1 M | 29.25 dB |
| **MDSR** | 2 | MDSR_baseline | 3.23 M | 34.63 dB |
| | | *MDSR | 7.95 M| 34.92 dB |
| | 3 | MDSR_baseline | | 30.94 dB |
| | | *MDSR | | 31.22 dB |
| | 4 | MDSR_baseline | | 28.97 dB |
| | | *MDSR | | 29.24 dB |

*Baseline models are in ``experiment/model``. Please download our final models from [here](https://cv.snu.ac.kr/research/EDSR/model_pytorch.tar) (542MB)
**We measured PSNR using DIV2K 0801 ~ 0900, RGB channels, without self-ensemble. (scale + 2) pixels from the image boundary are ignored.

You can evaluate your models with widely-used benchmark datasets:

[Set5 - Bevilacqua et al. BMVC 2012](http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html),

[Set14 - Zeyde et al. LNCS 2010](https://sites.google.com/site/romanzeyde/research-interests),

[B100 - Martin et al. ICCV 2001](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/),

[Urban100 - Huang et al. CVPR 2015](https://sites.google.com/site/jbhuang0604/publications/struct_sr).

For these datasets, we first convert the result images to YCbCr color space and evaluate PSNR on the Y channel only. You can download [benchmark datasets](https://cv.snu.ac.kr/research/EDSR/benchmark.tar) (250MB). Set ``--dir_data <where_benchmark_folder_located>`` to evaluate the EDSR and MDSR with the benchmarks.

You can download some results from [here](https://cv.snu.ac.kr/research/EDSR/result_image/edsr-results.tar).
The link contains **EDSR+_baseline_x4** and **EDSR+_x4**.
Otherwise, you can easily generate result images with ``demo.sh`` scripts.

## How to train EDSR and MDSR
We used [DIV2K](http://www.vision.ee.ethz.ch/%7Etimofter/publications/Agustsson-CVPRW-2017.pdf) dataset to train our model. Please download it from [here](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) (7.1GB).

Unpack the tar file to any place you want. Then, change the ```dir_data``` argument in ```src/option.py``` to the place where DIV2K images are located.

We recommend you to pre-process the images before training. This step will decode all **png** files and save them as binaries. Use ``--ext sep_reset`` argument on your first run. You can skip the decoding part and use saved binaries with ``--ext sep`` argument.

If you have enough RAM (>= 32GB), you can use ``--ext bin`` argument to pack all DIV2K images in one binary file.

You can train EDSR and MDSR by yourself. All scripts are provided in the ``src/demo.sh``. Note that EDSR (x3, x4) requires pre-trained EDSR (x2). You can ignore this constraint by removing ```--pre_train <x2 model>``` argument.

```bash
cd src       # You are now in */EDSR-PyTorch/src
sh demo.sh
```
