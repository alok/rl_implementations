#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import reduce

import torch
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch import Tensor, nn
from torch.autograd import Variable
from torch.nn import Parameter, Sequential
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, Dataset, TensorDataset

# function given len

epochs = 5


def gen_binary(num_strs: int = 10**5, length: int = 50):
    return torch.ByteTensor(1, num_strs, length).random_(0, 2).float()


def get_labels(t, length=50):
    return torch.LongTensor([xor(x) for x in t.view(-1, length)])


def xor(xs):
    parity = 0

    for x in xs:
        parity ^= int(x)

    return parity


class Model(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=50,
            hidden_size=5,
            num_layers=5,
        )

        self.fc1 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.rnn(x)[0]
        x = F.relu(x)
        x = self.fc1(x)
        x = softmax(x, dim=2)
        return x


if __name__ == '__main__':
    xs = gen_binary(10**2)
    ys = get_labels(xs)

    model = Model().cuda()

    loader = DataLoader(
        TensorDataset(xs, ys.view(1, -1)),
        pin_memory=True,
    )

    opt = optim.Adam(model.parameters())

    for epoch in range(epochs):
        for i, (x, y) in enumerate(loader):
            x, y = Variable(x).cuda(), Variable(y).cuda()

            pred = model(x)

            opt.zero_grad()
            loss = nn.functional.cross_entropy(pred, y)
            print(loss)

            loss.backward()
            opt.step()
