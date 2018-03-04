#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import functools
import itertools
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import typing
from functools import reduce
from logging import debug, info, log
from pathlib import Path
from typing import (Any, Callable, Dict, List, NamedTuple, NewType, Sequence,
                    TypeVar)

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from pudb import set_trace
from torch import Tensor, distributions, nn
from torch.autograd import Variable
from torch.nn import BatchNorm1d, Dropout, Linear, ReLU, Sequential, Softmax
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset

# TODO do state and action spaces have to be same for all ends?

# In paper, ant was dropped into 9 envs, but its S,A were same. No transfer
# learning yet.

State = Any
Action = Any
Timestep = int
Policy: Callable[[State], Action]

MAX_STEPS: Timestep = 1_000
SUBPOLICY_DURATION: Timestep = 2_00
NUM_POLICIES = 10

envs = ['Ant-v2']
envs = [gym.make(env) for env in envs]

# TODO check that all envs have same state and action spaces
# Start with random env
env = random.choice(envs)
S = int(np.prod(env.observation_space.shape))
A = int(np.prod(env.action_space.shape))
H = hidden_size = 50

# S->num_policies
# TODO try RNN?
master = Sequential(
    Linear(S, H),
    ReLU(),
    Linear(H, H),
    ReLU(),
    Linear(H, NUM_POLICIES),
    Softmax(dim=0),
)

# TODO try returning distribution directly in forward() method
def MasterPolicy(nn.Module):


# TODO return mean and variance for each
class Policy(nn.Module):
    def __init__(self):
        self.fc1 = Linear(S, H)
        self.fc2 = Linear(H, H)
        self.mean_head = Linear(H, A)
        self.std_head = Linear(H, A)

    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        a = F.relu(self.fc1(a))
        mean = self.mean_head(s)
        std = self.std_head(s)
        return mean, std


policies = []


def W(x: np.ndarray) -> Variable:
    """Wrap a numpy array into a torch variable."""
    return Variable(torch.from_numpy(s).float())


s = env.reset()

P = distributions.Categorical(master(W(s)))

# TODO subpolicies should output parameters of Gaussian of the right action shape

# first, implement policies and picking
# then, implement all the reward and optimization

# TODO think over what you're doing in terms of actually doing them, not in
# terms of what you need to write on a computer

# TODO turn loops into combinators so you can compose them
# TODO loop over sampled tasks
for t in itertools.count():
    # pick new subpolicy every X steps
    if t % SUBPOLICY_DURATION == 0:
        policy = policies[int(P.sample())]

    mean, std = policy(W(s))
    p = distributions.Normal(mean, std)
    a = p.sample()
    succ, r, done, _ = env.step(a)
    s = env.reset() if done else succ

    # TODO optimize each policy with something like PPO
    # loop through a whole rollout
    # TODO warmup for some number of timesteps
    # TODO

if __name__ == '__main__':
    pass
