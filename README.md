# EDSR-PyTorch
This repository is a PyTorch version of the paper **"Enhanced Deep Residual Networks for Single Image Super-Resolution"** from **CVPRW 2017**.
You can find the original code and more information from [here](https://github.com/LimBee/NTIRE2017).

If you find our work useful in your research or publication, please cite our work:

[1] Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee, **"Enhanced Deep Residual Networks for Single Image Super-Resolution,"** <i>2nd NTIRE: New Trends in Image Restoration and Enhancement workshop and challenge on image super-resolution in conjunction with **CVPR 2017**. </i> [[PDF](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Lim_Enhanced_Deep_Residual_CVPR_2017_paper.pdf)] [[arXiv](https://arxiv.org/abs/1707.02921)] [[Slide](http://cv.snu.ac.kr/research/EDSR/Presentation_v3(release).pptx)]
```
@InProceedings{Lim_2017_CVPR_Workshops,
  author = {Lim, Bee and Son, Sanghyun and Kim, Heewon and Nah, Seungjun and Lee, Kyoung Mu},
  title = {Enhanced Deep Residual Networks for Single Image Super-Resolution},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {July},
  year = {2017}
}
```
This repository provides some demo codes for reproducing all the results from the paper. (Include training scripts.)

Also, pre-trained model will be uploaded soon.

**Differences with Torch version**
* Codes are much more compact. (Removed all unnecessary parts.)
* Model sizes are smaller. (About half.)
* Training requires less memory.
* Test is faster.
* Python-based.

**Update log**
* Jan 04, 2018
  * Many parts are re-written. You cannot use previous scripts and models directly.
  * Pre-trained MDSR is temporarily disabled.
  * Training details are included.

## Dependencies
* Python (Tested with 3.6)
* PyTorch >= 0.2.0

## Code

Clone this repository into any place you want.
```bash
git clone https://github.com/thstkdgus35/EDSR-PyTorch
cd EDSR-PyTorch
```

## Quick start (Demo)
You can test our super-resolution algorithm with your own images.

Place your images in ```test``` folder. (like ```test/puppy.jpeg```)

Then, run the provided script in ```code``` folder.

Before you run the demo, please uncomment the appropriate line in ```demo.sh``` that you want to execute.
```bash
cd code       # You are now in */EDSR-PyTorch/code
sh demo.sh
```

You can find the result images from ```experiment/test_<modelName>``` folder.

We provide 3 pre-trained models (baseline ONLY, not full version.) till now. You can find the model from the ```experiment/model``` folder.

| Model | Scale | File Name | # ResBlocks | # Filters | # Parameters |
|  ---  |  ---  | ---       | ---         |---        |---           |
| **EDSR**| 4 | EDSR_baseline_x4.pt | 16 | 64 | 1.5M | 
| ~~**MDSR**~~| 2 + 3 + 4 | MDSR_baseline.pt | 16 | 64 | 3.2M |
| ~~**MDSR (JPEG)**~~*| 2 + 3 + 4 | MDSR_baseline_jpeg.pt | 16 | 64 | 3.2M |

*MDSR (JPEG) even reduces the JPEG artifact in output images. However, its DIV2K validation performance is slightly lower than the original MDSR.

## How to train EDSR and MDSR
We used [DIV2K](http://www.vision.ee.ethz.ch/%7Etimofter/publications/Agustsson-CVPRW-2017.pdf) dataset for training. Please download it from [here](http://cv.snu.ac.kr/research/EDSR/DIV2K.tar) (7.1GB).

Unpack the tar file to any place you want. Then, change the ```dir_data``` argument in ```code/option.py``` to the place where you unpack DIV2K images.

We recommend you to pre-process the images before training. This step will decode and collect all png files into one huge binary file. Use ```code/tools/png2binary.pt``` for this process.

If you do not have enough RAM (>= 16GB), change the ```ext``` argument in ```code/option.py``` to ```png```. However, each image in DIV2K is so large that disk access and decoding png files can be a bottleneck.

Training scripts are also included in ``demo.sh``. By uncommenting the appropriate line and executing the script, you can train EDSR and MDSR by yourself. Note that EDSR (x3, x4) requires pre-trained EDSR (x2). By removing ```--pre_train``` argument in the provided script, you can ignore this constraint.

```bash
cd code       # You are now in */EDSR-PyTorch/code
sh demo.sh
```