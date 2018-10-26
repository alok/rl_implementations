#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools

# 2 layers
import operator

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor, nn
from torch.autograd import Variable
from torch.nn import Linear, ReLU, Sequential, Softmax

N = 1000
BATCH_SIZE = 32


def gen_data(batch_size=BATCH_SIZE):
    xs = np.random.randint(low=0, high=2, size=(batch_size, 2))
    ys = np.array([operator.xor(*x) for x in xs])

    xs = Variable(torch.from_numpy(xs).float())
    ys = Variable(torch.from_numpy(ys).long())

    return xs, ys


def argmax(tensor, dim=1):
    return tensor.max(dim=dim)[1]


H = 8

model = Sequential(Linear(2, H), ReLU(), Linear(H, 2))

if __name__ == "__main__":

    criterion = F.cross_entropy
    optimizer = optim.Adam(model.parameters())
    # One batch is enough since XOR only has 4 possible values.
    xs, ys = gen_data()

    for i in range(N):
        preds = model(xs)
        loss = criterion(preds, ys)
        print(float(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
