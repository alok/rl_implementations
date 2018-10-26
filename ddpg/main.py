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

exp = Experiment(name="ddpg", debug=False, save_dir="logs")

parser = HyperOptArgumentParser(strategy="random_search")

parser.add_argument("--buffer_size", default=1_000_000, type=int)
parser.add_argument("--num_steps", default=100_000, type=int)
parser.add_argument("--hidden_size", default=50, type=int)

parser.add_opt_argument_list(
    "--batch_size",
    default=64,
    type=int,
    tunnable=True,
    options=[2 ** i for i in range(5, 9)],
)

parser.add_opt_argument_list(
    "--discount",
    default=0.995,
    type=float,
    tunnable=True,
    options=[0.9, 0.99, 0.995, 0.999],
)

parser.add_opt_argument_range(
    "--target_update",
    default=100,
    type=int,
    tunnable=True,
    start=10,
    end=100,
    nb_samples=5,
)

parser.add_opt_argument_range(
    "--noise_factor", default=1, type=float, start=0, end=1, nb_samples=5
)

hparams = parser.parse_args()

env = gym.make("Pendulum-v0")

state_size = int(np.prod(env.observation_space.shape))
action_size = int(np.prod(env.action_space.shape))

S, A = state_size, action_size
H = hparams.hidden_size

exp.add_metric_row({"S": state_size, "A": action_size, "H": hparams.hidden_size})


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
    td_estimate = rewards + ((1 - dones) * hparams.discount * Q_succ)
    return td_estimate.detach()


actor_opt = Adam(actor.parameters())
critic_opt = Adam(critic.parameters())

buffer = ReplayBuffer(hparams.buffer_size)
s, rews = np_to_var(env.reset()), []

for hparam in hparams.trials(5):
    exp.add_argparse_meta(hparam)
    for timestep in range(hparam.num_steps):
        noise = Normal(
            mean=Variable(torch.zeros(A)),
            std=hparam.noise_factor * Variable(torch.ones(A)),
        )

        if timestep % 1000 == 0:
            hparam.noise_factor /= 2

        a = actor(s) + noise.sample()
        succ, r, done, _ = env.step(a.data.numpy())
        succ = np_to_var(succ)
        buffer.append(Step(s, a, r, succ, done))
        rews.append(r)
        s = np_to_var(env.reset()) if done else succ
        if done:

            exp.add_metric_row({"Timestep": timestep + 1, "Loss": -sum(rews)})

            rews = []

        if len(buffer) >= hparam.batch_size:
            states, actions, rewards, succ_states, dones = format_batch(
                buffer.sample(hparam.batch_size)
            )

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

            if timestep % hparam.target_update == 0:
                # Hard update
                actor_target.load_state_dict(actor.state_dict())
                critic_target.load_state_dict(critic.state_dict())

    exp.save()
