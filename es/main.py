#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import sys
from pathlib import Path

import gym
import numpy as np
import ray
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch import Tensor, distributions, nn
from torch.autograd import Variable
from torch.nn import BatchNorm1d, Dropout, Linear, ReLU, Sequential, Softmax
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

# random mean and variance of action size
mean, std = Variable(Tensor([0.])), Variable(Tensor([0.]))

N = 10

# weights of policy should be random

def rollout(policy)

def rollout(mean, std):
    distrs = [distributions.Normal(mean, std) for _ in range(N)]
    mean = [distr.mean for distr in distrs]
    # TODO



# update mean at each iteration
# sample policies
# use top 20%'s mean to calculate new mean

# for iteration in itertools.count():

if __name__ == '__main__':
    pass
