#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random

import gym
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

BUFFER_SIZE = 10**6
BATCH_SIZE = 128
GAMMA = 0.99
NUM_EPISODES = 1000
TARGET_UPDATE = 100
EPS_START = 0.999
EPS_END = 0.01
EPS_DECAY = 200

env = gym.make('CartPole-v0')


def generate_epsilon(step):
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-step / EPS_DECAY)
    return eps_threshold


def eps_greedy(env, state, q_function, step):
    state = Variable(torch.from_numpy(state.astype(np.float32)), volatile=True)
    eps = generate_epsilon(step)
    if random.random() < eps:
        action = env.action_space.sample()
    else:
        action = int(q_function(state).data.max(0)[1])
    return action


def get_index(idx, batch):
    return [item[idx] for item in batch]


class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def append(self, transition):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition)
        else:
            idx = len(self.buffer) % self.buffer_size
            self.buffer[idx] = transition

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class Model(nn.Module):
    def __init__(self, env):
        super().__init__()

        S = self.state_size = int(np.product(env.observation_space.shape))
        A = self.action_size = env.action_space.n

        self.fc1 = nn.Linear(S, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, A)

        self.loss = nn.functional.mse_loss
        self.opt = torch.optim.Adam(self.parameters())

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train(self, batch, q_target):
        states = Variable(torch.Tensor(get_index(0, batch)))
        actions = Variable(torch.LongTensor(get_index(1, batch))[:, None])
        new_states = Variable(torch.Tensor(get_index(2, batch)), volatile=False)
        rewards = Variable(torch.Tensor(get_index(3, batch)))
        dones = Variable(torch.Tensor(get_index(4, batch)))

        predictions = self(states).gather(1, actions)
        _labels = q_target(new_states).detach()
        labels = GAMMA * (1 - dones) * _labels.max(1)[0] + rewards

        loss = self.loss(predictions, labels)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss


def main():
    step = 0
    buffer = ReplayBuffer(BUFFER_SIZE)
    # They don't have to be the same since they're essentially random in the
    # beginning.
    Q, Q_target = Model(env), Model(env)

    for episode in range(NUM_EPISODES):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = eps_greedy(env, state, Q_target, step)
            new_state, reward, done, _ = env.step(action)
            buffer.append([state, action, new_state, reward, done])
            state = new_state

            if len(buffer) >= BATCH_SIZE:
                batch = buffer.sample(BATCH_SIZE)
                loss = Q.train(batch, Q_target)

            if step % TARGET_UPDATE == 0 and step > 0:
                Q_target.load_state_dict(Q.state_dict())

            step += 1
            episode_reward += reward

        if len(buffer) >= BATCH_SIZE:
            print(
                f"Episode: {episode}, Reward: {episode_reward}, Loss: {float(loss.data)}, ε: {generate_epsilon(step)}"
            )


if __name__ == '__main__':
    main()
