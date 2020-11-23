import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
from torchvision import datasets, transforms

import os
import argparse
import pdb
import copy
import numpy as np
from torch.optim import lr_scheduler
import copy

from utils import *
from model import lenet
from fl_trainer import *

READ_CKPT=False

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--local_train_period', type=int, default=1,
                        help='number of local training epochs')
    parser.add_argument('--num_nets', type=int, default=3383,
                        help='number of totally available users')
    parser.add_argument('--part_nets_per_round', type=int, default=30,
                        help='number of participating clients per FL round')
    parser.add_argument('--fl_round', type=int, default=100,
                        help='total number of FL round to conduct')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device to set, can take the value of: cuda or cuda:x')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='dataset to use during the training process')
    parser.add_argument('--model', type=str, default='lenet',
                        help='model to use during the training process')
    parser.add_argument('--partition_strategy', type=str, default='homo',
                        help='data partition strategy: homo|hetero-dir')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {'num_workers': 30, 'pin_memory': False} if use_cuda else {}

    device = torch.device(args.device if use_cuda else "cpu")

    logger.info("Running Attack of the tails with args: {}".format(args))
    logger.info(device)
    logger.info('==> Building model..')

    torch.manual_seed(args.seed)
    criterion = nn.CrossEntropyLoss()

    net_dataidx_map = partition_data(
        args.dataset, './data', args.partition_strategy,
        args.num_nets, 0.5, args)

    # load dataset:
    test_loader = load_test_dataset(args=args)
    # READ_CKPT = False
    if READ_CKPT:
        if args.model == "lenet":
            net_avg = lenet.LeNet().to(device)
            with open("./checkpoint/mnist_lenet_10epoch.pt", "rb") as ckpt_file:
                ckpt_state_dict = torch.load(ckpt_file, map_location=device)
        else:
            pass
        net_avg.load_state_dict(ckpt_state_dict)
        logger.info("Loading checkpoint file successfully ...")
    else:
        if args.model == "lenet":
            net_avg = lenet.LeNet().to(device)

    logger.info("Test the model performance on the entire task before FL process ... ")

    test(net_avg, device, test_loader, criterion)

    # let's remain a copy of the global model for measuring the norm distance:
    vanilla_model = copy.deepcopy(net_avg)

    arguments = {
        "vanilla_model":vanilla_model,
        "test_loader":test_loader,
        "net_avg": net_avg,
        "net_dataidx_map": net_dataidx_map,
        "num_nets": args.num_nets,
        "dataset": args.dataset,
        "model": args.model,
        "part_nets_per_round": args.part_nets_per_round,
        "fl_round": args.fl_round,
        "local_training_period": args.local_train_period,  # 5 #1
        "lr":args.lr,
        "args_gamma": args.gamma,
        "batch_size": args.batch_size,
        "test_batch_size": args.test_batch_size,
        "log_interval": args.log_interval,
        "device": device,
        "partition_strategy":args.partition_strategy
    }

    fl_trainer = FederatedLearningTrainer(arguments=arguments)
    fl_trainer.run()

