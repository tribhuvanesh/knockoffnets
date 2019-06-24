import os.path as osp

from torchvision.datasets import MNIST as TVMNIST
from torchvision.datasets import EMNIST as TVEMNIST
from torchvision.datasets import FashionMNIST as TVFashionMNIST
from torchvision.datasets import KMNIST as TVKMNIST

import knockoff.config as cfg


class MNIST(TVMNIST):
    def __init__(self, train=True, transform=None, target_transform=None, download=True):
        root = osp.join(cfg.DATASET_ROOT, 'mnist')
        super().__init__(root, train, transform, target_transform, download)


class KMNIST(TVKMNIST):
    def __init__(self, train=True, transform=None, target_transform=None, download=True):
        root = osp.join(cfg.DATASET_ROOT, 'kmnist')
        super().__init__(root, train, transform, target_transform, download)


class EMNIST(TVEMNIST):
    def __init__(self, **kwargs):
        root = osp.join(cfg.DATASET_ROOT, 'emnist')
        super().__init__(root, split='balanced', download=True, **kwargs)
        # Images are transposed by default. Fix this.
        self.data = self.data.permute(0, 2, 1)


class EMNISTLetters(TVEMNIST):
    def __init__(self, **kwargs):
        root = osp.join(cfg.DATASET_ROOT, 'emnist')
        super().__init__(root, split='letters', download=True, **kwargs)
        # Images are transposed by default. Fix this.
        self.data = self.data.permute(0, 2, 1)


class FashionMNIST(TVFashionMNIST):
    def __init__(self, train=True, transform=None, target_transform=None, download=True):
        root = osp.join(cfg.DATASET_ROOT, 'mnist_fashion')
        super().__init__(root, train, transform, target_transform, download)
