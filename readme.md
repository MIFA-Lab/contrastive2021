# Contrastive Self-Supervised Learning on CIFAR-10

[![arXiv](https://img.shields.io/badge/arXiv-2111.00743-brightgreen)](https://arxiv.org/abs/2111.00743)
[![Platform](https://img.shields.io/badge/platform-PyTorch-orange)](https://pytorch.org/get-started/locally/)
[![Top Language](https://img.shields.io/github/languages/top/huang-research-group/contrastive2021)](https://github.com/huang-research-group/contrastive2021/search?l=python)
[![Latest Release](https://img.shields.io/github/v/release/huang-research-group/contrastive2021)](https://github.com/huang-research-group/contrastive2021/releases)

## Description

[Weiran Huang](https://www.weiranhuang.com), Mingyang Yi and Xuyang Zhao, "[Towards the Generalization of Contrastive Self-Supervised Learning](https://arxiv.org/abs/2111.00743)", arXiv:2111.00743, 2021.

This repository is used to verify how data augmentations will affect the performance of contrastive self-supervised learning algorithms.

## Supported Models

- SimCLR
- Barlow Twins
- MoCo
- SimSiam

## Supported Augmentations

- (a) Random Cropping
- (b) Random Gaussian Blur
- (c) Color Dropping (i.e., randomly convert images to grayscale)
- (d) Color Distortion
- (e) Random Horizontal Flipping

## Installation
```bash
python -m venv venv                 # create a virtual environment named venv
source venv/bin/activate            # activate the environment
pip install -r requirements.txt     # install the dependencies
```

Code is tested in the following environment:
- torch==1.4.0
- torchvision==0.5.0
- torchmetrics==0.4.0
- pytorch-lightning==1.3.8
- hydra-core==1.0.0
- lightly==1.0.8 **(important!)**

## Script and Sample Code

```console
├── contrastive2021
    ├── README.md                  // descriptions about the repo
    ├── requirements.txt           // dependencies
    ├── scripts
        ├── run_train_eval.sh      // shell script for training and evaluation
    ├── src
        ├── models.py              // models
        ├── util.py                // utilities
    ├── train_eval.py              // training and evaluation script
```

## Evaluation
KNN evaluation protocol. Code from [here](https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb).

## Results

### Richer data augmentations result in better performance

Usage: `python train_eval.py --model=simclr --epoch=800 --augs=abcde --num_runs=3`

| (a)  | (b)  | (c)  | (d)  | (e)  |    SimCLR    | Barlow Twins |     MoCo     |    SimSiam   |
| :--: | :--: | :--: | :--: | :--: | :----------: | :----------: | :----------: | :----------: |
|  ✓   |  ✓   |  ✓   |  ✓   |  ✓   | 89.76 ± 0.12 | 86.91 ± 0.09 | 90.12 ± 0.12 | 90.59 ± 0.11 |
|  ✓   |  ✓   |  ✓   |  ✓   |  ×   | 88.48 ± 0.22 | 85.38 ± 0.37 | 89.69 ± 0.11 | 89.34 ± 0.09 |
|  ✓   |  ✓   |  ✓   |  ×   |  ×   | 83.50 ± 0.14 | 82.00 ± 0.59 | 86.78 ± 0.07 | 85.38 ± 0.09 |
|  ✓   |  ✓   |  ×   |  ×   |  ×   | 63.23 ± 0.05 | 67.83 ± 0.94 | 75.12 ± 0.28 | 63.27 ± 0.30 |
|  ✓   |  ×   |  ×   |  ×   |  ×   | 62.74 ± 0.18 | 67.77 ± 0.69 | 74.94 ± 0.22 | 61.47 ± 0.74 |

### Stronger data augmentations result in better performance

Usage: `python train_eval.py --model=twins --epoch=800 --augs=color --color_strength=0.5 --num_runs=3`

| Color Distortion Strength |    SimCLR    | Barlow Twins |     MoCo     |    SimSiam   |
| :-----------------------: | :----------: | :----------: | :----------: | :----------: |
|             1             | 82.75 ± 0.24 | 82.58 ± 0.25 | 86.68 ± 0.05 | 82.50 ± 1.05 |
|            1/2            | 78.76 ± 0.18 | 81.88 ± 0.25 | 84.30 ± 0.14 | 81.80 ± 0.15 |
|            1/4            | 76.37 ± 0.11 | 79.64 ± 0.34 | 82.76 ± 0.09 | 78.80 ± 0.17 |
|            1/8            | 74.23 ± 0.16 | 77.96 ± 0.16 | 81.20 ± 0.12 | 76.09 ± 0.50 |


## Acknowledgement

This code is based on:

- [IgorSusmelj/barlowtwins](https://github.com/IgorSusmelj/barlowtwins)
- [lightly/imagenette_benchmark.py](https://github.com/lightly-ai/lightly/blob/v1.1.19/docs/source/getting_started/benchmarks/imagenette_benchmark.py)

## Citation

If you find our work useful in your research, please consider citing:

```
@misc{huang2021towards,
      title={Towards the Generalization of Contrastive Self-Supervised Learning}, 
      author={Weiran Huang and Mingyang Yi and Xuyang Zhao},
      year={2021},
      eprint={2111.00743},
      archivePrefix={arXiv}
}
```
