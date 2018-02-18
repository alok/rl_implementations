#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import operator
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor, nn
from torch.autograd import Variable
from torch.nn import (BatchNorm1d, BatchNorm2d, Conv2d, Dropout, Dropout2d,
                      Linear, MaxPool2d, Parameter, ReLU, Sequential, Softmax)
from torch.utils.data import DataLoader, Dataset, TensorDataset

NUM_SAMPLES = 100_000
INPUT_DIM = 1
NUM_CLASSES = 2
SEQ_LEN = 50
BATCH_SIZE = 2
OP = operator.xor


def foldr(arr: np.ndarray, op) -> np.ndarray:
    """Specific version of foldr that's only for Numpy arrays"""

    return np.fromiter(itertools.accumulate(arr, op), dtype=np.float32, count=len(arr))


# TODO generate minibatch in one shot


def gen_data(seq_len=SEQ_LEN, batch_size=BATCH_SIZE, input_dim=INPUT_DIM):
    xs = np.random.randint(low=0, high=2, size=(seq_len, batch_size, input_dim))
    # Have to do some weird slicing here to do the fold in one shot

    # I was stacking along the wrong axis
    ys = np.stack([foldr(xs[:, i, 0], OP) for i in range(xs.shape[1])], axis=-1)

    xs = Variable(torch.from_numpy(xs).float())
    ys = Variable(torch.from_numpy(ys).long())

    return xs.cuda(), ys.cuda()


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        H = 20

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=H,
            num_layers=3,
        )
        self.fc = nn.Linear(H, NUM_CLASSES)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


def argmax(tensor, dim=1):
    return tensor.max(dim=dim)[1]


# Hack to check if we've already trained a model (assumed to be a good one.
test_mode = Path('model.pth').exists()
train_mode = not test_mode

if test_mode:
    model = torch.load('model.pth')
    model.lstm.flatten_parameters()
else:
    model = Model().cuda()

if __name__ == '__main__':

    criterion = F.cross_entropy
    optimizer = optim.Adam(model.parameters())

    # TODO use variable len padded sequence

    for i in range(NUM_SAMPLES // BATCH_SIZE):

        seq_len = np.random.randint(low=1, high=50 + 1)
        batch_size = np.random.randint(low=2, high=32 + 1) if test_mode else BATCH_SIZE

        xs, ys = gen_data(
            seq_len=seq_len,
            batch_size=batch_size,
        )

        predictions = model(xs)

        loss = criterion(predictions.view(-1, NUM_CLASSES), ys.view(-1))

        if i % (NUM_SAMPLES // 200) == 0:
            print(f'Iteration: {i} | Batch size: {batch_size} | Loss: %.5f' % float(loss))

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if train_mode:
        torch.save(model, 'model.pth')
