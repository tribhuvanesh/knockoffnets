#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import os.path as osp
import os
from datetime import datetime
import json
import pickle
from collections import defaultdict as dd

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data as torch_data
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS, default_loader

import knockoff.config as cfg
from knockoff import datasets
import knockoff.utils.transforms as transform_utils
import knockoff.utils.model as model_utils
import knockoff.utils.utils as knockoff_utils

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


class TransferSet(ImageFolder):
    def __init__(self, samples, transform=None, target_transform=None):
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.transform = transform
        self.target_transform = target_transform


def train_adversary(model, trainset, out_path, batch_size=64, testset=None, device=None, num_workers=10, lr=0.1,
                    momentum=0.5, lr_step=30, lr_gamma=0.1, resume=None, epochs=100, log_interval=100,
                    weighted_loss=False, checkpoint_suffix='', **kwargs):
    if device is None:
        device = torch.device('cuda')
    if not osp.exists(out_path):
        knockoff_utils.create_dir(out_path)

    # Data loaders
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    if testset is not None:
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        test_loader = None

    # Optimizer
    criterion_train = model_utils.soft_cross_entropy
    criterion_test = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    start_epoch = 1
    best_test_acc, test_acc = -1., -1.

    # Resume if required
    if resume is not None:
        model_path = resume
        if osp.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            start_epoch = checkpoint['epoch']
            best_test_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

    model_out_path = osp.join(out_path, 'checkpoint{}.pth.tar'.format(checkpoint_suffix))
    for epoch in range(start_epoch, epochs + 1):
        scheduler.step(epoch)
        train_loss, train_acc = model_utils.train_step(model, train_loader, criterion_train, optimizer, epoch, device,
                                                       log_interval=log_interval)
        if test_loader is not None:
            test_loss, test_acc = model_utils.test_step(model, test_loader, criterion_test, device, epoch=epoch)

        if test_acc >= best_test_acc:
            state = {
                'epoch': epoch,
                'arch': model.__class__,
                'state_dict': model.state_dict(),
                'best_acc': test_acc,
                'optimizer': optimizer.state_dict(),
                'created_on': str(datetime.now()),
            }
            torch.save(state, model_out_path)

    return model


def main():
    parser = argparse.ArgumentParser(description='Train a model')
    # Required arguments
    parser.add_argument('model_dir', metavar='DIR', type=str, help='Directory containing transferset.pickle')
    parser.add_argument('model_arch', metavar='MODEL_ARCH', type=str, help='Model name')
    parser.add_argument('testdataset', metavar='DS_NAME', type=str, help='Name of test')
    parser.add_argument('--budgets', metavar='B', type=str,
                        help='Comma separated values of budgets. Knockoffs will be trained for each budget.')
    # Optional arguments
    parser.add_argument('-d', '--device', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--lr-step', type=int, default=60, metavar='N',
                        help='Step sizes for LR')
    parser.add_argument('--lr-gamma', type=float, default=0.1, metavar='N',
                        help='LR Decay Rate')
    parser.add_argument('-w', '--num_workers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained network', default=False)
    parser.add_argument('--weighted-loss', action='store_true', help='Use a weighted loss', default=False)
    args = parser.parse_args()
    params = vars(args)

    torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    params['device'] = device
    model_dir = params['model_dir']

    # ----------- Set up transferset
    transferset_path = osp.join(model_dir, 'transferset.pickle')
    with open(transferset_path, 'rb') as rf:
        transferset_samples = pickle.load(rf)
    num_classes = transferset_samples[0][1].size(0)
    print('=> found transfer set with {} samples, {} classes'.format(len(transferset_samples), num_classes))

    # ----------- Set up testset
    dataset_name = params['testdataset']
    valid_datasets = datasets.__dict__.keys()
    if dataset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    dataset = datasets.__dict__[dataset_name]
    testset = dataset(train=False, transform=transform_utils.DefaultTransforms.test_transform)
    if len(testset.classes) != num_classes:
        raise ValueError('# Transfer classes ({}) != # Testset classes ({})'.format(num_classes, len(testset.classes)))

    # ----------- Set up model
    model_name = params['model_arch']
    pretrained = params['pretrained']
    model = model_utils.get_net(model_name, n_output_classes=num_classes, pretrained=pretrained)
    model = model.to(device)

    # ----------- Train
    budgets = [int(b) for b in params['budgets'].split(',')]

    for b in budgets:
        print()
        print('=> Training at budget = {}'.format(b))

        np.random.seed(cfg.DEFAULT_SEED)
        torch.manual_seed(cfg.DEFAULT_SEED)
        torch.cuda.manual_seed(cfg.DEFAULT_SEED)

        transferset = TransferSet(transferset_samples[:b], transform=transform_utils.DefaultTransforms.train_transform)
        checkpoint_suffix = '.{}'.format(b)
        train_adversary(model, transferset, model_dir, testset=testset, checkpoint_suffix=checkpoint_suffix, **params)

    # Store arguments
    params['created_on'] = str(datetime.now())
    params_out_path = osp.join(model_dir, 'params_train.json')
    with open(params_out_path, 'w') as jf:
        json.dump(params, jf, indent=True)


if __name__ == '__main__':
    main()
