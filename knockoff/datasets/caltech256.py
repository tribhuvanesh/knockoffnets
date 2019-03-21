#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import os.path as osp
import os
from collections import defaultdict as dd

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


class Caltech256(ImageFolder):

    def __init__(self, train=True, transform=None, target_transform=None):
        root = osp.join(cfg.DATASET_ROOT, '256_ObjectCategories')
        if not osp.exists(root):
            raise ValueError('Dataset not found at {}. Please download it from {}.'.format(
                root, 'http://www.vision.caltech.edu/Image_Datasets/Caltech256/'
            ))

        # Initialize ImageFolder
        super().__init__(root=root, transform=transform, target_transform=target_transform)

        self._cleanup()
        self.ntest = 25  # Reserve these many examples per class for evaluation
        self.partition_to_idxs = self.get_partition_to_idxs()
        self.pruned_idxs = self.partition_to_idxs['train' if train else 'test']

        # Prune (self.imgs, self.samples to only include examples from the required train/test partition
        self.samples = [self.samples[i] for i in self.pruned_idxs]
        self.imgs = self.samples

        print('=> done loading {} ({}) with {} examples'.format(self.__class__.__name__, 'train' if train else 'test',
                                                                len(self.samples)))

    def _cleanup(self):
        # Remove examples belonging to class "clutter"
        clutter_idx = self.class_to_idx['257.clutter']
        self.samples = [s for s in self.samples if s[1] != clutter_idx]
        del self.class_to_idx['257.clutter']
        self.classes = self.classes[:-1]

    def get_partition_to_idxs(self):
        partition_to_idxs = {
            'train': [],
            'test': []
        }

        # Use this random seed to make partition consistent
        prev_state = np.random.get_state()
        np.random.seed(cfg.DS_SEED)

        # ----------------- Create mapping: classidx -> idx
        classidx_to_idxs = dd(list)
        for idx, s in enumerate(self.samples):
            classidx = s[1]
            classidx_to_idxs[classidx].append(idx)

        # Shuffle classidx_to_idx
        for classidx, idxs in classidx_to_idxs.items():
            np.random.shuffle(idxs)

        for classidx, idxs in classidx_to_idxs.items():
            partition_to_idxs['test'] += idxs[:self.ntest]     # A constant no. kept aside for evaluation
            partition_to_idxs['train'] += idxs[self.ntest:]    # Train on remaining

        # Revert randomness to original state
        np.random.set_state(prev_state)

        return partition_to_idxs

