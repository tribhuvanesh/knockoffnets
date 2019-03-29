import os.path as osp

from torchvision.datasets import MNIST as TVMNIST
from torchvision.datasets import EMNIST as TVEMNIST
from torchvision.datasets import FashionMNIST as TVFashionMNIST

import defenses.config as cfg


class MNIST(TVMNIST):
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(self, train=True, transform=None, target_transform=None, download=False):
        root = osp.join(cfg.DATASET_ROOT, 'mnist')
        super().__init__(root, train, transform, target_transform, download)


class EMNIST(TVEMNIST):
    def __init__(self, split='balanced', **kwargs):
        root = osp.join(cfg.DATASET_ROOT, 'emnist')
        super().__init__(root, split, **kwargs)


class FashionMNIST(TVFashionMNIST):
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    def __init__(self, train=True, transform=None, target_transform=None, download=False):
        root = osp.join(cfg.DATASET_ROOT, 'mnist_fashion')
        super().__init__(root, train, transform, target_transform, download)
