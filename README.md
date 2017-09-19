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
For some reason, we only provide demo script here.

However, you can train our model with this code if you make the script and prepare the dataset on your own.

## Dependencies
* Python (Tested with 3.6)
* PyTorch (**Supports 0.20 ONLY**)
## Code

Clone this repository into any place you want.
```bash
git clone https://github.com/thstkdgus35/EDSR-PyTorch
cd EDSR-PyTorch
```

## Quick Start (Demo)
You can test our super-resolution algorithm with your own images.

Place your images in ```test``` folder. (like ```test/puppy.jpeg```)

Then, run the provided script in ```code``` folder.

Before you run the demo, please uncomment the appropriate line in ```demo.sh``` that you want to execute.
```bash
cd code
sh demo.sh
```

You can find the result images from ```experiment/test_<modelName>``` folder.

We provide 3 pre-trained models (baseline ONLY, not full version.) till now. You can find the model from the ```experiment/model``` folder.

| Model | Scale | File Name | # ResBlocks | # Filters | # Parameters |
|  ---  |  ---  | ---       | ---         |---        |---           |
| **EDSR**| 4 | EDSR_baseline_x4.pt | 16 | 64 | 1.5M | 
| **MDSR**| 2 + 3 + 4 | MDSR_baseline.pt | 16 | 64 | 3.2M |
| **MDSR (JPEG)**| 2 + 3 + 4 | MDSR_baseline_jpeg.pt | 16 | 64 | 3.2M |

*MDSR (JPEG) even reduces the JPEG artifact in output images. However, its DIV2K validation performance is slightly lower than the original MDSR.