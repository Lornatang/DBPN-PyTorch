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

Modify the contents of the file as follows.

- line 25: `upscale_factor` change to the magnification you need to enlarge.
- line 27: `mode` change Set to valid mode.
- line 66: `model_path` change weight address after training.

## Train

Modify the contents of the file as follows.

- line 25: `upscale_factor` change to the magnification you need to enlarge.
- line 27: `mode` change Set to train mode.

If you want to load weights that you've trained before, modify the contents of the file as follows.

- line 41: `resume` change to `True`.
- line 42: `strict` Transfer learning is set to `False`, incremental learning is set to `True`.
- line 43: `start_epoch` change number of training iterations in the previous round.
- line 44: `resume_weight` the weight address that needs to be loaded.

## Result

Source of original paper results: https://arxiv.org/pdf/1904.05677.pdf

In the following table, the value in `()` indicates the result of the project, and `-` indicates no test.

| Dataset | Scale |       PSNR       |
|:-------:|:-----:|:----------------:|
|  Set5   |   2   | 38.08(**37.94**) |
|  Set5   |   4   |   32.65(**-**)   |
|  Set5   |   8   |   27.51(**-**)   |

Low Resolution / Super Resolution / High Resolution
<span align="center"><img src="assets/result.png"/></span>

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
