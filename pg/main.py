#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random

import gym
import numpy as np
import torch
import torch.optim as optim
from torch import Tensor, distributions
from torch.autograd import Variable
from torch.nn import Linear, ReLU, Sequential, Softmax

DISCOUNT = 0.99
NUM_EPISODES = 1000

env = gym.make('CartPole-v0')

S, A, H = int(np.prod(env.observation_space.shape)), int(env.action_space.n), 50

# for discrete action spaces only
actor = Sequential(
    Linear(S, H),
    ReLU(),
    Linear(H, H),
    ReLU(),
    Linear(H, A),
    Softmax(),
).cuda()

# Value function
critic = Sequential(
    Linear(S, H),
    ReLU(),
    Linear(H, H),
    ReLU(),
    Linear(H, 1),
).cuda()

opt = optim.Adam(list(actor.parameters()) + list(critic.parameters()))

def G(rewards, start=0, end=None):
    return sum(rewards[start:end])


if __name__ == '__main__':

    for episode in range(NUM_EPISODES):
        s, done = env.reset(), False
        states, rewards, log_probs = [], [], []

        while not done:
            s = Variable(torch.from_numpy(s).float()).cuda()
            p = distributions.Categorical(actor(s))
            a = p.sample()
            succ, r, done, _ = env.step(int(a.data))

            states.append(s)
            rewards.append(r)
            log_probs.append(p.log_prob(a))

            s = succ

        discounted_rewards = [pow(DISCOUNT, t) * r for t, r in enumerate(rewards)]
        cumulative_returns = [G(discounted_rewards, t) for t in range(len(discounted_rewards))]

        states = torch.stack(states).cuda()
        state_values = critic(states).view(-1)

        R = Variable(Tensor((cumulative_returns))).cuda()
        Adv = R - state_values  # Advantage estimator

        log_probs = torch.stack(log_probs).view(-1)

        loss = -((Adv @ log_probs) / len(rewards))

        opt.zero_grad()
        loss.backward()
        opt.step()

        print(f'ep: {episode}, R: {sum(rewards)}')

        if sum(rewards) > 200 - 5:
            print(30 * '-')
