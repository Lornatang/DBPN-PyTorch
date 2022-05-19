# DBPN-PyTorch

### Overview

This repository contains an op-for-op PyTorch reimplementation of [Deep Back-Projection Networks for Super-Resolution](https://arxiv.org/abs/1904.05677).

### Table of contents

- [DBPN-PyTorch](#dbpn-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [About Deep Back-Projection Networks for Super-Resolution](#about-deep-back-projection-networks-for-super-resolution)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
    - [Test](#test)
    - [Train](#train)
    - [Resume train](#resume-train)
    - [Result](#result)
    - [Credit](#credit)
        - [Deep Back-Projection Networks for Super-Resolution](#deep-back-projection-networks-for-super-resolution)

## About Deep Back-Projection Networks for Super-Resolution

If you're new to DBPN, here's an abstract straight from the paper:

Previous feed-forward architectures of recently proposed deep super-resolution networks learn the features of low-resolution inputs and the non-linear
mapping from those to a high-resolution output. However, this approach does not fully address the mutual dependencies of low- and high-resolution
images. We propose Deep Back-Projection Networks (DBPN), the winner of two image super-resolution challenges (NTIRE2018 and PIRM2018), that exploit
iterative up- and down-sampling layers. These layers are formed as a unit providing an error feedback mechanism for projection errors. We construct
mutually-connected up- and down-sampling units each of which represents different types of low- and high-resolution components. We also show that
extending this idea to demonstrate a new insight towards more efficient network design substantially, such as parameter sharing on the projection
module and transition layer on projection step. The experimental results yield superior results and in particular establishing new state-of-the-art
results across multiple data sets, especially for large scaling factors such as 8x.

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

Contains DIV2K, DIV8K, Flickr2K, OST, T91, Set5, Set14, BSDS100 and BSDS200, etc.

- [Google Driver](https://drive.google.com/drive/folders/1A6lzGeQrFMxPqJehK9s37ce-tPDj20mD?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1o-8Ty_7q6DiS3ykLU09IVg?pwd=llot)

## Test

Modify the contents of the `config.py` file as follows.

- line 33: `upscale_factor` change to `2`.
- line 35: `mode` change to `valid`.
- line 71: `model_path` change to `results/pretrained_models/DBPN-RES-MR64-3_x2-DIV2K-e604aa23.pth.tar`.

## Train

Modify the contents of the `config.py` file as follows.

- line 33: `upscale_factor` change to `2`.
- line 35: `mode` change to `train`.

## Resume train

If you want to load weights that you've trained before, modify the contents of the `config.py` file as follows.

- line 41: `resume` change to `samples/DBPN-RES-MR64-3_x2/epoch_xxx.pth.tar`.

## Result

Source of original paper results: https://arxiv.org/pdf/1904.05677.pdf

In the following table, the value in `()` indicates the result of the project, and `-` indicates no test.

| Method | Scale |      Set5 (PSNR/SSIM)      |      Set14(PSNR/SSIM)      |     BSD100(PSNR/SSIM)      |
|:------:|:-----:|:--------------------------:|:--------------------------:|:--------------------------:|
| D-DBPN |   2   | 38.27(**-**)/0.9616(**-**) | 34.28(**-**)/0.9231(**-**) | 32.47(**-**)/0.9032(**-**) |
| D-DBPN |   3   | 34.78(**-**)/0.9303(**-**) | 30.73(**-**)/0.8488(**-**) | 29.36(**-**)/0.8117(**-**) |
| D-DBPN |   4   | 32.63(**-**)/0.9002(**-**) | 28.94(**-**)/0.7900(**-**) | 27.83(**-**)/0.7444(**-**) |
| D-DBPN |   8   | 27.36(**-**)/0.7882(**-**) | 25.34(**-**)/0.6531(**-**) | 25.01(**-**)/0.6057(**-**) |

Low Resolution / Super Resolution / High Resolution
<span align="center"><img src="figure/result.png"/></span>

### Credit

#### Deep Back-Projection Networks for Super-Resolution

_Muhammad Haris, Greg Shakhnarovich, Norimichi Ukita_ <br>

**Abstract** <br>
Previous feed-forward architectures of recently proposed deep super-resolution networks learn the features of low-resolution inputs and the non-linear
mapping from those to a high-resolution output. However, this approach does not fully address the mutual dependencies of low- and high-resolution
images. We propose Deep Back-Projection Networks (DBPN), the winner of two image super-resolution challenges (NTIRE2018 and PIRM2018), that exploit
iterative up- and down-sampling layers. These layers are formed as a unit providing an error feedback mechanism for projection errors. We construct
mutually-connected up- and down-sampling units each of which represents different types of low- and high-resolution components. We also show that
extending this idea to demonstrate a new insight towards more efficient network design substantially, such as parameter sharing on the projection
module and transition layer on projection step. The experimental results yield superior results and in particular establishing new state-of-the-art
results across multiple data sets, especially for large scaling factors such as 8x.

[[Code (PyTorch) ]](https://github.com/alterzero/DBPN-Pytorch)
[[Code (Caffe) ]](https://github.com/alterzero/DBPN-caffe)
[[Paper]](https://arxiv.org/pdf/1904.05677)

```
@article{DBLP:journals/corr/abs-1904-05677,
  author    = {Muhammad Haris and
               Greg Shakhnarovich and
               Norimichi Ukita},
  title     = {Deep Back-Projection Networks for Single Image Super-resolution},
  journal   = {CoRR},
  volume    = {abs/1904.05677},
  year      = {2019},
  url       = {http://arxiv.org/abs/1904.05677},
  eprinttype = {arXiv},
  eprint    = {1904.05677},
  timestamp = {Thu, 25 Apr 2019 13:55:01 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1904-05677.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
