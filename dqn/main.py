#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch import nn
from torch.autograd import Variable as V
from torch.nn import Parameter as P
from torch.utils.data import DataLoader

from args import args
from exploration import decay_exploration, epsilon, epsilon_greedy
from model import Q

done = False
s = env.reset()  # TODO fold into rollout
i = 0  # TODO fold into rollout

optimizer = optim.Adam(Q.parameters(), lr=args.lr)

# TODO wrap data in dataset

while not done:

    epsilon = decay_exploration(i, epsilon)

    a = epsilon_greedy(s, epsilon=epsilon)

    r, succ, done, _ = env.step(a)
    replay_buffer.append([s, a, r, succ, done])

    s = succ

    if i % batch_size == 0 and i > 0:
        # TODO
        train(optimizer)

if __name__ == '__main__':
    pass
