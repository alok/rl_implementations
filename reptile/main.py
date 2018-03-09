#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.nn import Linear
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset

from utils import ParamDict

# TODO Use CUDA if available.
CUDA_AVAILABLE = torch.cuda.is_available()

INPUT_SIZE = 1
OUTPUT_SIZE = 1
HIDDEN_SIZE = 64
# BATCH_SIZE = 2**7
BATCH_SIZE = 1
EPOCHS = 1
META_LR = 1e-1  # copycat
META_EPOCHS = 30_000
META_BATCH_SIZE = 3

N = 10  # 10 ps on sine wave

Weights = ParamDict

criterion = F.mse_loss

Task = DataLoader

# TODO True loss (for evaluation) is integral over 50 evenly spaced points on
# -5,5.


def cuda(x):
    return x.cuda() if CUDA_AVAILABLE else x


def gen_task(input_size=INPUT_SIZE) -> Task:
    # amplitude
    a = np.random.uniform(low=0.1, high=5)  # amplitude
    b = np.random.uniform(low=0, high=2 * np.pi)  # phase

    # XXX Need size N,1 instead of N, to avoid auto conversion to DoubleTensor
    # later.
    x = np.random.uniform(low=-5, high=5, size=(N, input_size))  # num of samples
    y = a * np.sin(x + b)

    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        pin_memory=CUDA_AVAILABLE,  # TODO change to use cuda if available
    )

    return loader


class Model(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        I, O = INPUT_SIZE, OUTPUT_SIZE
        H = HIDDEN_SIZE
        self.fc1 = Linear(I, H)
        self.fc2 = Linear(H, H)
        self.out = Linear(H, O)

        # XXX this has to be after the weight initializations or else we get a
        # KeyError.
        if weights is not None:
            self.load_state_dict(weights)

    def forward(self, x):
        r = F.relu

        x = r(self.fc1(x))
        x = r(self.fc2(x))
        x = self.out(x)

        return x


def SGD(meta_weights: Weights, k: int) -> Weights:
    task: Task = gen_task()

    model = cuda(Model(meta_weights))

    model.train()

    opt = Adam(model.parameters())

    for epoch, (x, y) in zip(range(k), task):
        # x, y = next(task)
        x, y = cuda(Variable(x)), cuda(Variable(y))
        pred = model(x)
        loss = criterion(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    weights = ParamDict(model.state_dict())
    return weights


def evaluate(model: Model, task: Task) -> float:
    model.eval()

    # evaluate on all the data at once
    x, y = cuda(Variable(task.dataset.data_tensor)), cuda(Variable(task.dataset.target_tensor))
    preds = model(x)

    loss = criterion(preds, y)
    return float(loss)


if __name__ == '__main__':

    # need to put model on gpu first for tensors to have the right type
    meta_weights = ParamDict(cuda(Model()).state_dict())

    for i in range(META_EPOCHS):

        weights = [SGD(meta_weights, k=EPOCHS) for _ in range(META_BATCH_SIZE)]
        # the mul by 0 is to get a paramdict of the right size as the start value for summation.
        # TODO implement custom optimizer that makes this work with Adam easily
        meta_weights = meta_weights + (META_LR / len(weights)) * sum(
            (w - meta_weights for w in weights), 0 * weights[0]
        )

    ###########################################################################
    # Few shot learning
    ###########################################################################

    # train model

    model = cuda(Model(meta_weights))
    model.train()  # set train mode
    opt = Adam(model.parameters())

    task = gen_task()

    # 3 epochs to train
    for i, (x, y) in zip(range(3), task):
        x, y = cuda(Variable(x)), cuda(Variable(y))

        preds = model(x)
        loss = criterion(preds, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # test model
    print(evaluate(model, task))
