#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random

import torch
from torch.autograd import Variable

from args import args
from env import env
from model import Q

# S  -> A

epsilon = args.exploration_rate


def epsilon_greedy(s, Q=Q, epsilon=epsilon):
    # torch expects FloatTensors, so we use `.float()`
    s = torch.from_numpy(s).float()
    s = Variable(s).cuda()

    if random.random() <= epsilon:
        a = env.action_space.sample()
    else:
        a = int(Q(s).max(0)[1])

    return a


def decay_exploration(i, epsilon=epsilon):
    if i > 0 and i % args.batch_size == 0:
        epsilon = max(0.01, epsilon / 2)
    return epsilon


if __name__ == '__main__':
    pass
    print(epsilon_greedy(env.observation_space.sample()))
