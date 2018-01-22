#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym

A = 2  # actions are discrete  (0 or 1)
S = 4  # states are (4,) arr
gamma = discount = .999

env = gym.make('CartPole-v0')
