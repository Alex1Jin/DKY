import os
import argparse
import json
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import logging
import torchvision
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

from itertools import product
import math
import copy
import time
import logging
import pickle
import random
import logging

from dataset import *

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def partition_data(dataset, datadir, partition, n_nets, alpha, args):
    if dataset == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
        n_train = X_train.shape[0]
    elif dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
        n_train = X_train.shape[0]

    if partition == "homo":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero-dir":
        min_size = 0
        K = 10
        N = y_train.shape[0]
        net_dataidx_map = {}

        while (min_size < 10) or (dataset == 'mnist' and min_size < 100):
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    return net_dataidx_map

def load_mnist_data(datadir):
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = MNIST_truncated(datadir, train=True, download=True, transform=transform)
    mnist_test_ds = MNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)


def load_test_dataset(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {'num_workers': 30, 'pin_memory': False} if use_cuda else {}
    if args.dataset == 'mnist':
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                           ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    return test_loader

def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    if dataset in ('mnist', 'emnist', 'cifar10'):
        if dataset == 'mnist':
            dl_obj = MNIST_truncated

            transform_train = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])

            transform_test = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])

        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

        train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True)
        test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    return train_dl, test_dl