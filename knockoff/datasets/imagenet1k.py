#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import os.path as osp

import numpy as np
from torchvision.datasets import ImageFolder

import knockoff.config as cfg

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


class ImageNet1k(ImageFolder):
    test_frac = 0.2

    def __init__(self, train=True, transform=None, target_transform=None):
        root = osp.join(cfg.DATASET_ROOT, 'ILSVRC2012')
        if not osp.exists(root):
            raise ValueError('Dataset not found at {}. Please download it from {}.'.format(
                root, 'http://image-net.org/download-images'
            ))

        # Initialize ImageFolder
        super().__init__(root=osp.join(root, 'training_imgs'), transform=transform,
                         target_transform=target_transform)
        self.root = root

        self.partition_to_idxs = self.get_partition_to_idxs()
        self.pruned_idxs = self.partition_to_idxs['train' if train else 'test']

        # Prune (self.imgs, self.samples to only include examples from the required train/test partition
        self.samples = [self.samples[i] for i in self.pruned_idxs]
        self.imgs = self.samples

        print('=> done loading {} ({}) with {} examples'.format(self.__class__.__name__, 'train' if train else 'test',
                                                                len(self.samples)))

    def get_partition_to_idxs(self):
        partition_to_idxs = {
            'train': [],
            'test': []
        }

        # Note: we perform a 80-20 split of imagenet training
        # While this is not necessary, this is to simply to keep it consistent with the paper
        prev_state = np.random.get_state()
        np.random.seed(cfg.DS_SEED)

        idxs = np.arange(len(self.samples))
        n_test = int(self.test_frac * len(idxs))
        test_idxs = np.random.choice(idxs, replace=False, size=n_test).tolist()
        train_idxs = list(set(idxs) - set(test_idxs))

        partition_to_idxs['train'] = train_idxs
        partition_to_idxs['test'] = test_idxs

        np.random.set_state(prev_state)

        return partition_to_idxs
