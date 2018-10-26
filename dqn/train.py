#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch import FloatTensor, Tensor, nn
from torch.autograd import Variable
from torch.nn import Parameter as P
from torch.utils.data import DataLoader, Dataset

from args import args
from replay_buffer import replay_buffer

# TODO dataset and data loader

criterion = torch.nn.SmoothL1Loss()


class Replay(Dataset):
    def __getitem__(self, index):
        return replay_buffer[index]

    def __len__(self):
        return len(replay_buffer)


def train(buffer, Q):
    sample = random.sample(buffer, args.batch_size)

    # TODO implement sampling and tensorize everything

    states = torch.FloatTensor([i[0] for i in sample])
    actions = [i[1] for i in sample]
    rewards = [i[2] for i in sample]
    succs = torch.FloatTensor([i[3] for i in sample])
    dones = [i[4] for i in sample]

    # only states and succ states are passed to torch
    # TODO add dones
    states = Variable(states, volatile=True).cuda()
    succs = Variable(succs, volatile=True).cuda()

    # actions = sample[:, 1]
    # rewards = sample[:, 2]

    # should *not* be a tensor since this is used to check for equality
    # TODO implement as boolean mask
    # dones = [i[4] for i in sample]
    # dones = torch.from_numpy(sample[:, 4])

    td_estimates = Q(states)
    best_succ_qval, _argmaxes = Q(succs).max(1)

    for s, (_, a, r, succ, done) in enumerate(
        zip(states, actions, rewards, best_succ_qval, dones)
    ):
        if done:
            td_estimates[s, a] = r
        else:
            td_estimates[s, a] = r + args.discount_rate * succ

    states.volatile, td_estimates.volatile = False, False
    return states, td_estimates


if __name__ == "__main__":
    pass
