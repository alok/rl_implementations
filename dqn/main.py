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
from torch.autograd import Variable

from args import args
from env import env
from exploration import decay_exploration, epsilon, epsilon_greedy
from model import Q
from replay_buffer import replay_buffer
from train import criterion, train

optimizer = optim.Adam(Q.parameters(), lr=args.lr)

# TODO wrap data in dataset

for i in range(args.iterations):

    done = False
    s = env.reset()  # TODO fold into rollout

    while not done:

        epsilon = decay_exploration(i, epsilon)

        a = epsilon_greedy(s, epsilon=epsilon)


        succ, r, done, _ = env.step(a)
        replay_buffer.append([s, a, r, succ, done])

        s = succ

        if i % args.batch_size == 0 and i > 0 and len(replay_buffer) >= args.batch_size:

            # TODO cuda and var
            state, val = train(replay_buffer, Q)
            y = Q(state)

            optimizer.zero_grad()
            loss = criterion(y, val)
            loss.backward()
            print(loss.data[0])
            for param in Q.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()

if __name__ == '__main__':
    pass
