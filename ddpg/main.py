#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from logging import log

import gym
import numpy as np
import torch
import torch.nn.functional as F
from test_tube import Experiment, HyperOptArgumentParser
from torch import Tensor, nn
from torch.autograd import Variable
from torch.distributions import Normal
from torch.nn import BatchNorm1d, Dropout, Linear, ReLU
from torch.nn.functional import relu
from torch.optim import Adam

from utils import ParamDict, ReplayBuffer, Step, np_to_var

exp = Experiment(
    name='ddpg',
    debug=False,
    save_dir='logs',
)

parser = HyperOptArgumentParser(strategy='random_search')
parser.add_argument('--num_steps', default=100_000, type=int, tunnable=False)
parser.add_argument('--batch_size', default=64, type=int, tunnable=False)
parser.add_argument('--num_steps', default=100_000, type=int, tunnable=False)
parser.add_argument('--discount', default=0.995, type=float)
parser.add_argument('--target_update', default=100, type=int)
parser.add_argument('--soft_update_factor', default=.01, type=float)
parser.add_argument('--hidden_size', default=50, type=int)
parser.add_argument('--buffer_size', default=1_000_000, type=int)
parser.add_argument('--noise_factor', default=1, type=float)

args = parser.parse_args()

env = gym.make('Pendulum-v0')

state_size = int(np.prod(env.observation_space.shape))
action_size = int(np.prod(env.action_space.shape))

S, A = state_size, action_size
H = args.hidden_size

exp.add_metric_row({
    'S': state_size,
    'A': action_size,
    'H': hidden_size,
})


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(S + A, H)
        self.fc2 = Linear(H, H)
        self.out = Linear(H, 1)

    def forward(self, s, a):
        q = torch.cat((s, a), dim=1)
        q = relu(self.fc1(q))
        q = relu(self.fc2(q))
        q = self.out(q)
        return q


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(S, H)
        self.fc2 = Linear(H, H)
        self.out = Linear(H, A)

    def forward(self, s):
        s = relu(self.fc1(s))
        s = relu(self.fc2(s))
        s = self.out(s)
        return s


actor, critic = Actor(), Critic()

actor_target, critic_target = Actor(), Critic()
actor_target.load_state_dict(actor.state_dict())
critic_target.load_state_dict(critic.state_dict())


def format_batch(batch):
    """Get everything into PyTorch."""

    states = torch.stack([step.state for step in batch])
    actions = torch.stack([step.action for step in batch])
    rewards = Variable(Tensor([step.reward for step in batch]))
    succ_states = torch.stack([step.succ_state for step in batch])
    dones = Variable(Tensor([step.done for step in batch]))

    return states, actions, rewards, succ_states, dones


def get_critic_train_data(succ_states, rewards, dones):
    # r + Q(s, pi(s'))
    Q_succ = critic_target(succ_states, actor_target(succ_states)).squeeze()
    td_estimate = rewards + ((1 - dones) * DISCOUNT * Q_succ)
    return td_estimate.detach()


actor_opt = Adam(actor.parameters())
critic_opt = Adam(critic.parameters())

buffer = ReplayBuffer(BUFFER_SIZE)
s, rews = np_to_var(env.reset()), []

for timestep in range(NUM_STEPS):
    noise = Normal(mean=Variable(torch.zeros(A)), std=NOISE_FACTOR * Variable(torch.ones(A)))

    if timestep % 1000 == 0:
        NOISE_FACTOR /= 2

    a = actor(s) + noise.sample()
    succ, r, done, _ = env.step(a.data.numpy())
    succ = np_to_var(succ)
    buffer.append(Step(s, a, r, succ, done))
    rews.append(r)
    s = np_to_var(env.reset()) if done else succ
    if done:
        logging.info(f'step:{timestep + 1} | Loss: {-sum(rews)}')
        rews = []

    if len(buffer) >= BATCH_SIZE:
        states, actions, rewards, succ_states, dones = format_batch(buffer.sample(BATCH_SIZE))

        td_estims = get_critic_train_data(succ_states, rewards, dones)

        critic_preds = critic(states, actions.detach())

        critic_opt.zero_grad()
        critic_loss = F.smooth_l1_loss(critic_preds, td_estims)

        critic_loss.backward()
        critic_opt.step()

        actor_opt.zero_grad()
        actor_loss = -critic(states, actor(states)).mean()

        actor_loss.backward()
        actor_opt.step()

        if timestep % TARGET_UPDATE == 0:
            # Hard update
            actor_target.load_state_dict(actor.state_dict())
            critic_target.load_state_dict(critic.state_dict())

exp.save()
