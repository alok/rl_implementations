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
from torch import Tensor, distributions, nn
from torch.autograd import Variable
from torch.nn import (BatchNorm1d, BatchNorm2d, Conv2d, Dropout, Dropout2d,
                      Linear, MaxPool2d, Parameter, ReLU, Sequential, Softmax)
from torch.utils.data import DataLoader, Dataset, TensorDataset

N = 1
NUM_CLASSES = 2
SEQ_LEN = 50
BATCH_SIZE = 32


def foldr(arr: np.ndarray, op) -> np.ndarray:
    """Specific version of foldr that's only for Numpy arrays"""
    return np.fromiter(accumulate(arr, op), dtype=np.float32, count=len(arr))


# TODO generate minibatch in one shot


def gen_data():
    xs = Variable(torch.from_numpy(np.random.randint(low=0, high=2, size=(SEQ_LEN, N))).float())
    ys = Variable(torch.from_numpy(np.array([foldr(x, operator.xor) for x in xs], dtype=np.int64)))
    return xs, ys


class rnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.h = Variable(torch.Tensor([0]))
        self.fc1 = Linear(2, 1)
        self.out = nn.functional.sigmoid

    def forward(self, xs):
        h = self.h

        for i, x in enumerate(xs.view(-1)):
            h = self.fc1(torch.cat((x, h)))

        y = self.out(h)
        return y


model = nn.LSTM(
    input_size=1,
    hidden_size=2,
    num_layers=3,
)

if __name__ == '__main__':

    criterion = nn.functional.cross_entropy
    optimizer = optim.Adam(model.parameters())

    while True:
        xs, ys = gen_data()

        predictions, hiddens = model(xs[:, None])
        # predictions = nn.functional.softmax(predictions)

        loss = criterion(predictions.view(-1, NUM_CLASSES), ys.squeeze())
        print(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
