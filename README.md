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
We used the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset  to train our models and the wildely used benchmark datasets Set5, Set14, B100 and Urban100 to test the models.

## Pre-trained models
The models used for the experiments can be found in [experiment](/experiment) under <name_model>/model_best.pt

## Demo
You can test the SR resolution alogrithms with your images. Place your image in the ```test```folder. The network supports .png and .jpeg files
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
If you wish you can change the scale, 3 and 4 are mainly used, as well as the n_resblock to have deeper network, many other options are possible and are listed in ```option.py```

To test the performance of a pre-trained network run the following lines in your terminal
```bash
cd src       # You are now in */EDSR-PyTorch/src
python main.py --test_only --pre_train ../experiment/<name_network>/model/model_best.pt --data_test <benchmark dataset> --save <...> --save_results --reset
```


# Project Documentation

## Introduction

Due to the rise of mobile devices like drones and smartphones, camera sensors must be made smaller and more lightweight. This leads to a decrease in image resolution and quality. This is where single image super resolution, which aims at recovering a high resolution (HR) images from the degraded low resolution (LR) images, comes in handy. In this work we look at the performance of super resolution convolutional networks (SRCNN) which are trained on a large dataset of paired LR and HR images and aims to minimize a loss function that grades the performance of the super resolution (SR) images, the outputs of the network, compared to the HR images and then test them on different datasets. The resulting networks can then be used in mobile devices to increase the quality of the LR images.


## Baseline

Our project is based on the [EDSR-PyTorch](https://github.com/thstkdgus35/EDSR-PyTorch) repository which allready includes Loss-functions, some basic blocks, dataloaders and training and testing functions. In addition we use a [function](https://github.com/ofsoundof/dhp/blob/master/restoration/data/div2ksub.py) from an other repository which transforms the images into .pt files. We added some new simpler models to the repository.


## Background

There are two general aproaches for solving the super resolution problem. First you can interpolate your low resolution image using for example bicubic interpolation, and then run a convolutional neural network (CNN) on the interpolated images. Or you can add an upsampling block at the end of the CNN, which makes the network smaller and therefore faster. The EDSR model, which we also tested to see how the dataloaoding, training, testing, etc. works, uses the second approach. In this project both approches are tested, while using the first approach only for the simpler networks.


## Method & experiment

For the training of the networks we use HR images, which we degrade with a simplified model including blurring, downsampling with bicubic interpolation and noise, to get paired training data (LR, HR). Then we train our network on a large datasets of images to try do the inverse process and reconstituing a HR image. In fact the network is trained to exctract high frequencies information from a low-frequency input.
In the next lines we will look at the results of different networks on the benchmark datasets. Different techniques such as global residual connection, residual blocks or channel attention were used to try to improve the performance of the networks. We will also look at the importance of the network depth and width for the performance.


**Networks with a bicubic interpolated input**

![](/figs/basic_blocks.png)

The first SRCNN network we implemented was a three layers CNN with layer consisiting of a convolution followed by a ReLU activation function. After that we added a global residual connection (grc) to the network, as in the image, and then augmented the number of layers from 3 to 8. The input of the network is a LR image interpolated with a bicubic interpolation to have to size of the HR image.

![](/figs/task1_3_comparison.png)

We see that the global residual connection increase a little the performance of the network, but it is mainly the increase of the depth of the network that causes a ~1.5 dB increase in the peak signal to noise ratio (PSNR). 


 **Upsampling block added at the end of the network**

 Comparing the same 8 layers network with residual connection, one with bicubic interpolated input images and the other with an upsampling block at the end of the network we clearly see ...
 
 
 
From this point, all our networks add functionalities to the 8-layers SRCNN network with an upsampling block, thus we will always compare ou results with this network.

 **batch normalization and residual connection block** 
 In this part, we analyse the influence of batch normalization and residual blocks to the network.
 ![](/figs/task4_6_comparison.png)
 
 Adding batch normalization to the network, that mean normalizing the data of the batches after each convolutional layer we noticed a significant increase of the PSNR but the visual results are not a lot better. Indeed the benefit of batch normalization for SRCNN is not that clear yet. 
 
 On the other hand, replacing our blocks constituted of a convolutional layer followed by an actctivation function by residual blocks described in the picture below increase the PSNR a little less than the batch normalization, but the  visual result is more satisfying for the human eye.
 
 ![](/figs/resblock_v2.png)
 

 **Leaky ReLU and PreLU**
 
Here we look at the effects of the activation function, as default activation function we used ReLU(x) = max(x,0) and we compare the performance of of RelU with LeakyReLU and the parametric ReLu, PReLU. 
 
![alt-text-1](/figs/leakyrelu.png)![alt-text-2](/figs/prelu.png)

The parameter 'a' in the PReLU activation is a learnable parameter for each feature map.
LeakyReLU and PReLU are used to solve the problem of the "dying ReLU", due to the zero value and thus zero slope of the negative side of ReLU so once a neuron gets negative it "dies" and does not contribute anymore to the imporvement of the network. LeakyReLU and PReLU try to solve this problem by having a small positiv gradient on the negative side and thus they will not "kill" neurons.
 
 ![](/figs/leakyvsprelu.png)
 
 First of all, notice that the results here are not as good as in the others experiments. This can be easily explained, due to an unfortunate error the patch size in this experiment is 196 and not 96 as in the other experiment, thus the parameters of the network are not updated as often as usual resulting in a less optimized network and therefore results not as good as in the other experiments.
 
 But we can still compare the performance of the networks and we clearly see that the results for leakyReLU and PReLU are better than for a classic ReLU activation function. The visual result is better too, however we would have expected the PReLU achieve better results than the leakyReLU due to the additional learing parameter 'a' but this does not seem to be the case.

 **effect of deeper and wider network **
 
 ![](/figs/task8_comparison.png)
 ""
 TODO
 ""
 ![](/figs/task8_32_64.PNG)

 **loss function: L1 vs L2 vs perceptual loss**
 
 ![](/figs/task9_11_comparison.png)
 ""
 TODO
 ""

results in a table 

| Model | task | description |  SET5 PSNR | Set14 PSNR | Urban100 PSNR | B100 PSNR |
|  ---  |  ---  | ---       | ---  | ---        | ---  | ---        |
| **Bicubic** | 1 | 3 layers scale 2 | 36.504 dB | 32.346 dB | 29.275 dB | 31.212 dB
| | | 3 layers scale 3  | 32.366 dB | 29.053 dB | 25.858 dB | 28.186 dB
| | |  3 layers scale 4 | 30.041 dB | 27.263 dB | 24.126 dB | 26.682 dB
| | 2 | 3 layers with grc | 36.722 dB | 32.475 dB | 29.536 dB | 31.318 dB
| | 3 | 8 layers with grc | 37.432 dB | 33.049 dB | 30.752 dB | 31.809 dB
| **SRCNN** | 4 | upsampling | 37.335 dB | 32.933 dB | 30.383 dB | 31.697 dB |
| | 5 | batch normalisation | 37.636 dB | 33.175 dB | 31.119 dB | 31.929 dB
| | 6 | residual connections | 37.629 dB | 33.208 dB | 31.083 dB |  31.938 dB |
| | 7 | leakyReLU | 28.151 dB | 26.931 dB | 25.078 dB | 31.945 dB |
| | | PreLU | 27.652 dB | 26.728 dB | 24.922 dB | 31.944 dB |
| | 8 | 16 layers | 37.686 dB | 33.256 dB | 31.348 dB | 32.012 dB |
| | | 32 layers | 36.918 dB | 32.615 dB | 29.682 dB | 31.478 dB |
| | | batchsize 96 | 37.712 dB | 33.273 dB | 31.333 dB | 31.997 dB |
| | | batchsize 128 | 37.760 dB | 33.331 dB | 31.478 dB | 32.023 dB |
| | 9 | L2(mse) | 37.533 dB | 33.160 dB | 31.054 dB | 31.902 dB |
| | 10 | channel attention | 30.119 dB | 28.830 dB | 26.311 dB | 32.215 dB |
| | 11 | perceptual loss |  34.653 dB | 31.025 dB | 28.457 dB | 29.958 dB |




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
