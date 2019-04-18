#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import os.path as osp
from torchvision.datasets import ImageFolder

import knockoff.config as cfg

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


class TinyImageNet200(ImageFolder):
    """
    Dataset for TinyImageNet200

    Note: the directory structure slightly varies from original
    To get there, run these two commands:
    - From within tiny-images-200 directory
        for dr in train/*; do
            echo $dr;
            mv $dr/images/* $dr/;
            rmdir $dr/images;
        done
    - From within tiny-images-200/val directory
         while read -r fname label remainder; do
            mkdir -p val2/$label;
            mv images/$fname val2/$label/;
        done < val_annotations.txt

    """

    def __init__(self, train=True, transform=None, target_transform=None):
        root = osp.join(cfg.DATASET_ROOT, 'tiny-imagenet-200')
        if not osp.exists(root):
            raise ValueError('Dataset not found at {}. Please download it from {}.'.format(
                root, 'https://tiny-imagenet.herokuapp.com'
            ))

        # Initialize ImageFolder
        _root = osp.join(root, 'train' if train else 'val')
        super().__init__(root=_root, transform=transform,
                         target_transform=target_transform)
        self.root = root

        print('=> done loading {} ({}) with {} examples'.format(self.__class__.__name__, 'train' if train else 'test',
                                                                len(self.samples)))

        self._load_meta()

    def _load_meta(self):
        """Replace class names (synsets) with more descriptive labels"""
        # Load mapping
        synset_to_desc = dict()
        fpath = osp.join(self.root, 'words.txt')
        with open(fpath, 'r') as rf:
            for line in rf:
                synset, desc = line.strip().split(maxsplit=1)
                synset_to_desc[synset] = desc

        # Replace
        for i in range(len(self.classes)):
            self.classes[i] = synset_to_desc[self.classes[i]]
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
