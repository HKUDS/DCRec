# DCRec

This is the PyTorch implementation for our paper **Debiased Contrastive Learning for Sequential Recommendation**, accpeted by **WWW'23**. The code is built on the [RecBole](https://github.com/RUCAIBox/RecBole) library, implemented by [@yuh-yang](https://github.com/yuh-yang).

## Citation
```
@inproceedings{dcrec2023,
  author    = {Yang, Yuhao and
               Huang, Chao and
               Xia, Lianghao and
               Huang, Chunzhen and
               Luo, Da and
               Lin Kangyi},
  title     = {Debiased Contrastive Learning for Sequential Recommendation},
  booktitle = {Proceedings of the ACM Web Conference 2023},
  year      = {2023},
}
```

## Introduction
As self-supervised learning (SSL) has proven to be effective in the field of recommender systems, researchers have sought to leverage this paradigm by introducing contrastive learning tasks into sequential recommendation models. To incorporate supplementary SSL signals, one line is to utilize various data augmentations on sequences or item features to enforce agreement between the augmented views. Other methods identify semantically positive pairs for sequences or items. Although these methods have shown significant improvements in performances, we believe that existing methods have not adequately addressed the inherent popularity bias in such data augmentation.
