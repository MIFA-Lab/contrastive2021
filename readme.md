# Contrastive Self-Supervised Learning on CIFAR-10

## Paper

"[Towards the Generalization of Contrastive Self-Supervised Learning](https://arxiv.org/abs/2111.00743)",
[Weiran Huang](https://www.weiranhuang.com), Mingyang Yi and Xuyang Zhao, arXiv:2111.00743, 2021.

The most critical argument we made in our paper is that the **quality of data augmentation** exhibits great impact on the quality of contrastive-learned encoder. The data augmentation with sharper intra-class concentration enables the model to have better generalization on downstream tasks. We verify it through a variety of experiments in this repository. 

## Supported methods

- SimCLR
- Barlow Twins
- MoCo
- SimSiam

## Installation
`pip install -r requirement.txt`

## Dependencies
- torch==1.4.0
- torchvision==0.5.0
- pytorch-lightning==1.3.8
- lightly==1.0.8 **(important!)**

## Evaluation
KNN evaluation protocol. Code from [here](https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb).

## Results

### ResNet-18 trained by SimCLR and Barlow Twins over various data augmentation combinations.

Example: `python main.py --model=twins --epoch=800 --batch=512 --round=3 --augs=abcde`

| (a)  | (b)  | (c)  | (d)  | (e)  |    SimCLR    | Barlow Twins |
| :--: | :--: | :--: | :--: | :--: | :----------: | :----------: |
|  ✓   |  ✓   |  ✓   |  ✓   |  ✓   | 89.92 ± 0.05 | 83.93 ± 0.57 |
|  ✓   |  ✓   |  ✓   |  ✓   |  ×   | 88.41 ± 0.11 | 83.37 ± 0.43 |
|  ✓   |  ✓   |  ✓   |  ×   |  ×   | 83.62 ± 0.19 | 73.70 ± 0.99 |
|  ✓   |  ✓   |  ×   |  ×   |  ×   | 62.91 ± 0.25 | 49.56 ± 0.11 |
|  ✓   |  ×   |  ×   |  ×   |  ×   | 62.37 ± 0.09 | 48.54 ± 0.29 |

Augmentation operations include:

(a) random cropping with a scaling factor chosen in [0.08, 1.0]; 

(b) random Gaussian blur with a probability 0.5; 

(c) color dropping (i.e., randomly convert images to grayscale with 0.2 probability for each image); 

(d) color distortion with a probability of 0.8 and with strength of [0.4, 0.4, 0.4, 0.1]; 

(e) random horizontal flipping with a probability of 0.5.  



### ResNet18 trained by SimCLR and Barlow Twins over various color distortion strengths.

Example: `python main.py --model=simclr --epoch=800 --batch=512 --round=3 --augs=color --strength=1`

| Color Distortion Strength |    SimCLR    | Barlow Twins |
| :-----------------------: | :----------: | :----------: |
|            1/8            | 73.60 ± 0.11 | 61.13 ± 2.81 |
|            1/4            | 76.25 ± 0.16 | 68.30 ± 0.15 |
|            1/2            | 78.49 ± 0.09 | 72.76 ± 1.50 |
|             1             | 82.64 ± 0.57 | 78.79 ± 0.54 |

## Acknowledgement

This code is based on:

- [IgorSusmelj/barlowtwins](https://github.com/IgorSusmelj/barlowtwins)
- [lightly/imagenette_benchmark.py](https://github.com/lightly-ai/lightly/blob/master/docs/source/getting_started/benchmarks/imagenette_benchmark.py)

## Citation

If you find our work useful in your research, please consider citing:

```
@misc{huang2021generalization,
      title={Towards the Generalization of Contrastive Self-Supervised Learning}, 
      author={Weiran Huang and Mingyang Yi and Xuyang Zhao},
      year={2021},
      eprint={2111.00743},
      archivePrefix={arXiv}
}
```

