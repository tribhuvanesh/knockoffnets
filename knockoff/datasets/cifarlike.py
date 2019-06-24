import sys
import os
import os.path as osp

from torchvision.datasets import CIFAR10 as TVCIFAR10
from torchvision.datasets import CIFAR100 as TVCIFAR100
from torchvision.datasets import SVHN as TVSVHN

import knockoff.config as cfg
from torchvision.datasets.utils import check_integrity
import pickle


class CIFAR10(TVCIFAR10):
    base_folder = 'cifar-10-batches-py'
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, train=True, transform=None, target_transform=None, download=False):
        root = osp.join(cfg.DATASET_ROOT, 'cifar10')
        super().__init__(root, train, transform, target_transform, download)

    def get_image(self, index):
        return self.data[index]


class CIFAR100(TVCIFAR100):
    base_folder = 'cifar-100-python'
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }

    def __init__(self, train=True, transform=None, target_transform=None, download=False):
        root = osp.join(cfg.DATASET_ROOT, 'cifar100')
        super().__init__(root, train, transform, target_transform, download)

    def get_image(self, index):
        return self.data[index]


class SVHN(TVSVHN):
    def __init__(self, train=True, transform=None, target_transform=None, download=False):
        root = osp.join(cfg.DATASET_ROOT, 'svhn')
        # split argument should be one of {‘train’, ‘test’, ‘extra’}
        if isinstance(train, bool):
            split = 'train' if train else 'test'
        else:
            split = train
        super().__init__(root, split, transform, target_transform, download)