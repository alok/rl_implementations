#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import random
from typing import Any

import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions, nn
from torch.autograd import Variable
from torch.nn import Linear
from torch.optim import Adam

from utils import ReplayBuffer, Step, np_to_var

# In paper, ant was dropped into 9 envs, but its S,A were same. No transfer
# learning yet.

State = Any
Action = Any
Timestep = int

MAX_STEPS: Timestep = 1_000
SUBPOLICY_DURATION: Timestep = 200
NUM_POLICIES = 10
MAX_BUFFER_SIZE = 10**6
BATCH_SIZE = 128

env_names = ['Ant-v2']


class MasterPolicy(nn.Module):
    """Returns categorical distribution over subpolicies."""

    def __init__(self, state_size, hidden_size, output_size=NUM_POLICIES):
        super().__init__()
        S, H, O = state_size, hidden_size, output_size
        self.fc1 = Linear(S, H)
        self.fc2 = Linear(H, H)
        self.out = Linear(H, O)

    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        s = F.softmax(self.out(s), dim=0)
        P = distributions.Categorical(s)
        return P


class Policy(nn.Module):
    def __init__(self, state_size, hidden_size, action_size):
        super().__init__()
        S, H, A = state_size, hidden_size, action_size
        self.fc1 = Linear(S, H)
        self.fc2 = Linear(H, H)
        self.mean_head = Linear(H, A)
        self.std_head = Linear(H, A)

    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        mean = self.mean_head(s)
        std = self.std_head(s)
        return mean, std


# first, implement policies and picking
# then, implement all the reward and optimization

# TODO turn loops into combinators so you can compose them
# TODO loop over sampled tasks

# TODO optimize each policy with something like PPO
# loop through a whole rollout
# TODO warmup for some number of timesteps


def rollout(env, start_state: State, policy, buffer, num_steps=SUBPOLICY_DURATION) -> None:
    s = start_state

    for i in range(num_steps):
        mean, std = policy(np_to_var(s))
        p = distributions.Normal(mean, std)
        a = p.sample()
        succ, r, done, _ = env.step(a.data.numpy())
        env.done = done
        buffer.append(Step(s, a, r, succ, done))
        s = np_to_var(env.reset()) if done else succ


if __name__ == '__main__':

    envs = [gym.make(env) for env in env_names]
    # TODO assert that all envs have same state and action spaces
    # Start with random env
    for env in envs:
        env.done = False
    env = random.choice(envs)

    S = int(np.prod(env.observation_space.shape))
    A = int(np.prod(env.action_space.shape))
    H = hidden_size = 50

    policies = [Policy(S, H, A) for _ in range(NUM_POLICIES)]

    buffer = ReplayBuffer(MAX_BUFFER_SIZE)

    s = env.reset()

    master = MasterPolicy(S, H, NUM_POLICIES)
    P = master(np_to_var(s))

    for t in itertools.count():
        # pick new subpolicy every X steps
        if t % SUBPOLICY_DURATION == 0:
            policy = policies[int(P.sample())]
        rollout(env, s, policy, buffer, SUBPOLICY_DURATION)
