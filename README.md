# EDSR-PyTorch
This repository is a PyTorch version of the paper **"Enhanced Deep Residual Networks for Single Image Super-Resolution"** from **CVPRW 2017**.
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
This repository provides some demo codes for reproducing all the results from the paper. (Include training scripts.)

Also, pre-trained model will be uploaded soon.

**Differences between Torch version**
* Codes are much more compact. (Removed all unnecessary parts.)
* Models are smaller. (About half in their sizes.)
* Slightly better performances.
* Training requires less memory.
* Test is faster.
* Python-based.

**Recent updates**
* Mar 11, 2018
  * Fixed some typos in the code and script.
  * Now --ext img is default setting. Although we recommend you to use --ext bin when training, please use --ext img when you use --test_only.
  * Skip_batch operation is implemented. Use --skip_threshold argument to skip the batch that you want to ignore. Although this function is not exactly same with that of Torch7 version, it will work as you expected.

## Dependencies
* Python (Tested with 3.6)
* PyTorch >= 0.3.1

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

We provide some pre-trained models. (Not full version, baseline ONLY.) You can find the model from the ```experiment/model```.
Also, these models have better performance than the original Torch7 models.

| Model | Scale | File name | ResBlocks | Filters | Parameters | **PSNR (PyTorch)** | PSNR (Torch7) |
|  ---  |  ---  | ---       | ---       | ---     | ---        | ---  | ---         |
| **EDSR** | 2 | EDSR_baseline_x2.pt | 16 | 64 | 1.5M | 34.61 | 34.55 |
| **EDSR** | 3 | EDSR_baseline_x3.pt | 16 | 64 | 1.5M | 30.92 | 30.90 |
| **EDSR** | 4 | EDSR_baseline_x4.pt | 16 | 64 | 1.5M | 28.95 | 28.94 |
| **MDSR** | 2 | MDSR_baseline.pt | 16 | 64 | 3.2M | 34.63 | 34.60 |
| | 3 | | | | | 30.94 | 30.91 |
| | 4 | | | | | 28.97 | 28.95 |

*We measured PSNR using DIV2K 0801 ~ 0900

## How to train EDSR and MDSR
We used [DIV2K](http://www.vision.ee.ethz.ch/%7Etimofter/publications/Agustsson-CVPRW-2017.pdf) dataset for training. Please download it from [here](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) (7.1GB).

Unpack the tar file to any place you want. Then, change the ```dir_data``` argument in ```code/option.py``` to the place where you unpack DIV2K images.

We recommend you to pre-process the images before training. This step will decode and collect all png files into one huge binary file. Use ```code/tools/png2binary.py``` for this process.

If you do not have enough RAM (>= 16GB), change the ```ext``` argument in ```code/option.py``` to ```png```. However, each image in DIV2K is so large that disk access and decoding png files can be a bottleneck.

Training scripts are also included in ``demo.sh``. By uncommenting the appropriate line and executing the script, you can train EDSR and MDSR by yourself. Note that EDSR (x3, x4) requires pre-trained EDSR (x2). By removing ```--pre_train``` argument in the provided script, you can ignore this constraint.

```bash
cd code       # You are now in */EDSR-PyTorch/code
sh demo.sh
```

**Update log**
* Jan 04, 2018
  * Many parts are re-written. You cannot use previous scripts and models directly.
  * Pre-trained MDSR is temporarily disabled.
  * Training details are included.

* Jan 09, 2018
  * Missing files are included (```code/data/MyImage.py```).
  * Some links are fixed.

* Jan 16, 2018
  * Memory efficient forward function is implemented.
  * Add --chop_forward argument to your script to enable it.
  * Basically, this function first split a large image to small patches. Those images are merged after super-resolution. I checked this function with 12GB memory, 4000 x 2000 input image in scale 4. (Therefore, the output will be 16000 x 8000.)

* Feb 21, 2018
  * Fixed the problem when loading pre-trained multi-gpu model.
  * Added pre-trained scale 2 baseline model.
  * This code now only saves the best-performing model by default. For MDSR, 'the best' can be ambiguous. Use --save_models argument to save all the intermediate models.
  * PyTorch 0.3.1 changed their implementation of DataLoader function. Therefore, I also changed my implementation of MSDataLoader. You can find it on feature/dataloader branch.

* Feb 23, 2018
  * Now PyTorch 0.3.1 is default. Use legacy/0.3.0 branch if you use the old version.
   
  * With a new ``code/data/DIV2K.py`` code, one can easily create new data class for super-resolution.
  * New binary data pack. (Please remove the ``DIV2K_decoded`` folder from your dataset if you have.)
  * With ``--ext bin``, this code will automatically generates and saves the binary data pack that corresponds to previous ``DIV2K_decoded``. (This requires huge RAM (~45GB, Swap can be used.), so please be careful.)
  * If you cannot make the binary pack, just use the default setting (``--ext img``).

  * Fixed a bug that PSNR in the log and PSNR calculated from the saved images does not match.
  * Now saved images have better quality! (PSNR is ~0.1dB higher than the original code.)
  * Added performance comparison between Torch7 model and PyTorch models.

* Mar 5, 2018
  * All baseline models are uploaded.
  * Now supports half-precision at test time. Use ``--precision half``  to enable it. This does not degrade the output images.

