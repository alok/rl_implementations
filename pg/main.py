#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch import Tensor, distributions, nn
from torch.autograd import Variable
from torch.nn import Parameter as P
from torch.nn import Sequential
from torch.utils.data import DataLoader, Dataset, TensorDataset

from lib.replay_buffer import ReplayBuffer

env = gym.make('CartPole-v0')

S, A = int(np.prod(env.observation_space.shape)), int(env.action_space.n)

# for discrete action spaces only
policy = nn.Sequential(
    nn.Linear(S, 50),
    nn.ReLU(),
    nn.Linear(50, 50),
    nn.ReLU(),
    nn.Linear(50, A),
    nn.Softmax(),
)

opt = optim.Adam(policy.parameters())



BUFFER_SIZE = 10**6
BATCH_SIZE = 128
GAMMA = 0.99
NUM_EPISODES = 1000
TARGET_UPDATE = 100
EPS_START = 0.999
EPS_END = 0.01
EPS_DECAY = 200


def eps_greedy(env, s, policy):

    if random.random < eps:
        a = env.action_space.sample()
    else:
        s = Variable(torch.from_numpy(s).float())
        p = distributions.Categorical(policy(s))

        a = p.sample()

    return a


if __name__ == '__main__':
    buffer = ReplayBuffer(BUFFER_SIZE)

    for episode in range(NUM_EPISODES):
        s: np.ndarray = env.reset()
        done = False

        while not done:
            #
            a = eps_greedy(s)
            succ, r, done, _ = env.step(a)


        R = sum(rewards)

            # TODO append data and train minibatch
