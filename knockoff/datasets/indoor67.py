#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import os.path as osp
import os

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision.datasets.folder import ImageFolder, default_loader

import knockoff.config as cfg

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


class Indoor67(ImageFolder):
    def __init__(self, train=True, transform=None, target_transform=None):
        root = osp.join(cfg.DATASET_ROOT, 'indoor')
        if not osp.exists(root):
            raise ValueError('Dataset not found at {}. Please download it from {}.'.format(
                root, 'http://web.mit.edu/torralba/www/indoor.html'
            ))

        # Initialize ImageFolder
        super().__init__(root=osp.join(root, 'Images'), transform=transform,
                         target_transform=target_transform)
        self.root = root

        self.partition_to_idxs = self.get_partition_to_idxs()
        self.pruned_idxs = self.partition_to_idxs['train' if train else 'test']

        # Prune (self.imgs, self.samples to only include examples from the required train/test partition
        self.samples = [self.samples[i] for i in self.pruned_idxs]
        self.imgs = self.samples

    def get_partition_to_idxs(self):
        partition_to_idxs = {
            'train': [],
            'test': []
        }

        # ----------------- Load list of train images
        test_images = set()
        with open(osp.join(self.root, 'TestImages.txt')) as f:
            for line in f:
                test_images.add(line.strip())

        for idx, (filepath, _) in enumerate(self.samples):
            filepath = filepath.replace(osp.join(self.root, 'Images') + '/', '')
            if filepath in test_images:
                partition_to_idxs['test'].append(idx)
            else:
                partition_to_idxs['train'].append(idx)

        return partition_to_idxs
