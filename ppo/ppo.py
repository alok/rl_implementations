#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A version of PPO that gathers data in parallel."""

import argparse
from copy import deepcopy
from typing import NamedTuple

import gym
import numpy as np
import ray
import torch
import torch.nn.functional as F
from torch import nn, tensor
from torch.distributions.categorical import Categorical
from torch.nn import Linear
from torch.optim import Adam


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--epochs", type=int, default=10 ** 4)
parser.add_argument("-d", "--discount", type=float, default=0.999)
parser.add_argument("-e", "--env", type=str, default="CartPole-v0")
parser.add_argument("-c", "--clip", type=float, default=0.15)
parser.add_argument("--num-workers", "-w", type=int, default=2 ** 4)
args = parser.parse_args()

GAMMA = args.discount
EPOCHS = args.epochs
NUM_WORKERS = args.num_workers

env = gym.make(args.env)

state_size = int(np.prod(env.observation_space.shape))
action_size = int(env.action_space.n)


# Coupling policy and value function didn't help much.
class Policy(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=50):
        super().__init__()

        S, A, H = state_size, action_size, hidden_size

        self.h1 = Linear(S, H)
        self.h2 = Linear(H, H)
        self.out = Linear(H, A)

    def forward(self, s):
        s = F.relu(self.h1(s))
        s = F.relu(self.h2(s))
        s = F.softmax(self.out(s), dim=None)
        p = Categorical(s)
        return p


class VF(nn.Module):
    def __init__(self, state_size, hidden_size=50):
        super().__init__()

        S, H = state_size, hidden_size

        self.h1 = Linear(S, H)
        self.h2 = Linear(H, H)
        self.out = Linear(H, 1)

    def forward(self, s):
        s = F.relu(self.h1(s))
        s = F.relu(self.h2(s))
        return self.out(s)


class Rollout(NamedTuple):
    states: torch.Tensor
    actions: int
    rewards: float


class Model:
    def __init__(self, state_size, action_size, hidden_size=50):
        self.policy = Policy(state_size, action_size, hidden_size)
        self.vf = VF(state_size, hidden_size)

    def __call__(self, s):
        return self.policy(s)


@ray.remote
def gather_data(old_model):
    """Execute one rollout."""
    env = gym.make(args.env)
    s, done = env.reset(), False

    rollout = Rollout(states=[], actions=[], rewards=[])

    while not done:
        s = torch.from_numpy(s).float()
        p = old_model(s)
        a = p.sample()

        with torch.no_grad():
            succ, r, done, _ = env.step(a.numpy())

        rollout.states.append(s)
        rollout.actions.append(a)
        rollout.rewards.append(r)

        s = succ

    return rollout


def train(model, old_model, data) -> float:
    """Statefully trains on batch of data."""
    states, actions, rewards = data

    states = torch.stack(states)
    actions = torch.stack(actions)

    discounted_rewards = [GAMMA ** t * r for t, r in enumerate(rewards)]
    cumulative_returns = tensor(
        [sum(discounted_rewards[t:]) for t, _ in enumerate(discounted_rewards)]
    )

    state_values = model.vf(states).flatten()

    adv = cumulative_returns - state_values
    vf_loss = F.mse_loss(state_values, cumulative_returns)

    # Old model is copied anyway, so no updates to it are necessary.
    with torch.no_grad():
        log_probs_old = old_model(states).log_prob(actions)
    log_probs_curr = model(states).log_prob(actions)

    ratio = torch.exp(log_probs_curr - log_probs_old)

    policy_loss = -torch.min(
        (adv * ratio).mean(), (adv * ratio.clamp(1 - args.clip, 1 + args.clip)).mean()
    )

    loss = policy_loss + vf_loss

    optimizer = Adam(list(model.policy.parameters()) + list(model.vf.parameters()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


if __name__ == "__main__":

    ray.init()
    model = Model(state_size, action_size)
    old_model = deepcopy(model)

    for epoch in range(EPOCHS):

        rollouts = ray.get([gather_data.remote(old_model) for _ in range(NUM_WORKERS)])

        for data in rollouts:
            print(f"Epoch: {epoch}, Total Reward: {int(sum(data.rewards))}")
            train(model, old_model, data)

        old_model = deepcopy(model)
