#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym

env = gym.make('CartPole-v0')

A = env.action_space.n  # actions are discrete  (0 or 1)
S = 4  # states are (4,) arr
