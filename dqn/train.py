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
from torch.utils.data import DataLoader, Dataset

from args import args
from replay_buffer import replay_buffer

# TODO dataset and data loader


class Replay(Dataset):
    def __getitem__(self, index):
        return replay_buffer[index]

    def __len__(self):
        return len(replay_buffer)


def train(buffer):

    # TODO implement sampling and tensorize everything

    states = NotImplemented
    succs = NotImplemented
    actions = NotImplemented
    dones = NotImplemented
    rewards = NotImplemented
    q_s = Q(states)
    q_succ = Q(succs)

    for i, (s, a, r, succ, done) in enumerate(zip(states, actions, rewards, succs, dones)):
        if done:
            q_s[a] = r
        else:
            q_s[a] = r + args.gamma * q_succ[a]
        pass


loss = torch.nn.MSELoss()

if __name__ == '__main__':
    pass
