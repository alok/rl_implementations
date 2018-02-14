#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import operator
import os
import random
import re
import shutil
import subprocess
import sys
from functools import reduce
from itertools import accumulate
from logging import debug, info, log
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from pudb import set_trace
from torch import Tensor, distributions, nn
from torch.autograd import Variable
from torch.nn import (BatchNorm1d, BatchNorm2d, Conv2d, Dropout, Dropout2d,
                      Linear, MaxPool2d, Parameter, ReLU, Sequential, Softmax)
from torch.utils.data import DataLoader, Dataset, TensorDataset

xs = np.random.randint(low=0, high=2, size=(50, 50))


def foldr(arr: np.ndarray, op) -> np.ndarray:
    """Specific version of foldr that's only for Numpy arrays"""
    return np.fromiter(accumulate(arr, op), dtype=np.float32, count=len(arr))


ys = np.array([foldr(row, operator.xor) for row in xs])

assert xs.shape == ys.shape

xs, ys = Variable(torch.from_numpy(xs).float()), Variable(torch.from_numpy(ys))


rnn = nn.RNN(
    input_size=50,
    hidden_size=50,
    num_layers=3,
)

if __name__ == '__main__':
    rnn(xs,ys)

