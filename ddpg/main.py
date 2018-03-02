#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import random
from copy import deepcopy
from typing import Any, NamedTuple

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor, nn
from torch.autograd import Variable
from torch.distributions import Normal
from torch.nn import BatchNorm1d, Dropout, Linear, ReLU, Sequential
from torch.nn.functional import relu

env = gym.make('Pendulum-v0')

state_size = int(np.prod(env.observation_space.shape))
action_size = int(np.prod(env.action_space.shape))
S, A = state_size, action_size
H = hidden_size = 50

NUM_EPISODES = 10**4

BUFFER_SIZE = 1_000_000
BATCH_SIZE = 32
DISCOUNT = 0.99
TARGET_UPDATE = 100


class Step(NamedTuple):
    """One step of a rollout."""
    state: Variable
    action: Variable
    succ_state: Variable
    reward: float
    done: bool


class ReplayBuffer(list):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size

    def append(self, transition):

        if len(self) < self.buffer_size:
            super().append(transition)
        else:
            idx = len(self) % self.buffer_size
            self[idx] = transition

    def sample(self, batch_size):
        return random.sample(self, batch_size)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(S + A, H)
        self.fc2 = Linear(H, H)
        self.out = Linear(H, 1)

    def forward(self, s, a):
        x = torch.cat((s, a), dim=1)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = self.out(x)
        return x


actor = Sequential(
    # BatchNorm1d(S),
    Linear(S, H),
    ReLU(),
    Linear(H, H),
    ReLU(),
    Linear(H, A),
)

critic = Critic()


def format_batch(batch):
    """Get everything into PyTorch."""

    states = torch.stack([step.state for step in batch])
    actions = torch.stack([step.action for step in batch])
    rewards = Variable(torch.Tensor([0 if step.done else step.reward for step in batch]))
    succ_states = torch.stack([step.succ_state for step in batch])
    # dones = Variable(torch.Tensor([step.done for step in batch]))

    # return states, actions, rewards, succ_states, dones
    return states, actions, rewards, succ_states


# def get_critic_train_data(succ_states, rewards):
#     # r + Q(s, pi(s'))
#     succ_states.volatile=True
#     rewards.volatile=True
#     Q_succ = critic_target(succ_states, actor_target(succ_states)).squeeze()
#     td_estimate = rewards + DISCOUNT * Q_succ
#     td_estimate.volatile=False
#     succ_states.volatile=False
#     rewards.volatile=False
#     return td_estimate

def get_critic_train_data(succ_states, rewards):
    # r + Q(s, pi(s'))
    Q_succ = critic_target(succ_states, actor_target(succ_states)).squeeze()
    td_estimate = (rewards + DISCOUNT * Q_succ).detach()
    return td_estimate


actor_target, critic_target = deepcopy(actor), deepcopy(critic)

noise = Normal(mean=Variable(torch.zeros(A)), std=Variable(torch.ones(A))*1e-3)

buffer = ReplayBuffer(BUFFER_SIZE)

actor_opt, critic_opt = optim.Adam(actor.parameters()), optim.Adam(critic.parameters())

for iteration in range(NUM_EPISODES):

    s, done = Variable(torch.from_numpy(env.reset()).float()), False

    rews=[]
    while not done:
        # TODO decrease noise over time
        a = actor(s) + noise.sample()
        succ, r, done, _ = env.step(a.data.numpy())
        succ = Variable(torch.from_numpy(succ).float())
        buffer.append(Step(s, a, succ, r, done))
        rews.append(r)
        s = succ

    states, actions, rewards, succ_states = format_batch(buffer.sample(BATCH_SIZE))

    td_estims = get_critic_train_data(succ_states, rewards)
    critic_preds = critic(states, actions)
    critic_opt.zero_grad()
    critic_loss = F.smooth_l1_loss(critic_preds, td_estims)

    critic_loss.backward()
    # TODO Why am I getting an error about retaining the graph?
    critic_opt.step()

    actor_loss = -torch.mean(critic(states, actor(states)))

    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()

    print(sum(rews))

    if iteration % TARGET_UPDATE == 0:
        # TODO soft target updates
        actor_target, critic_target = deepcopy(actor), deepcopy(critic)
