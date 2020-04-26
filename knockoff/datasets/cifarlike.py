import sys
import os
import os.path as osp

from torchvision.datasets.folder import ImageFolder
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


class TinyImagesSubset(ImageFolder):
    """
    A 800K subset of the 80M TinyImages data consisting of 32x32 pixel images from the internet. 
    Note: that the dataset is unlabeled.
    """
    def __init__(self, train=True, transform=None, target_transform=None):
        root = osp.join(cfg.DATASET_ROOT, 'tiny-images-subset')
        if not osp.exists(root):
            raise ValueError('Dataset not found at {}. Please download it from {}.'.format(
                root, 'https://github.com/Silent-Zebra/tiny-images-subset'
            ))

        # Initialize ImageFolder
        fold = 'train' if train else 'test'
        super().__init__(root=osp.join(root, fold), transform=transform,
                         target_transform=target_transform)
        self.root = root

        print('=> done loading {} ({}) with {} examples'.format(self.__class__.__name__, 'train' if train else 'test',
                                                                len(self.samples)))