#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import random
import re
import shutil
import subprocess
import sys
from logging import debug, info, log
from pathlib import Path

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch import Tensor, nn
from torch.autograd import Variable as V
from torch.nn import Parameter as P
from torch.utils.data import DataLoader, Dataset

env = gym.make('CartPole-v0')
iterations = 1000

for iteration in range(iterations):
    s: np.ndarray = env.reset()
    done = False

    while not done:
        #
        probs = pi(Variable(torch.from_numpy(s)))
        a = probs.sample()
        succ, r, done, _ = env.step(a)



        # TODO append data and train minibatch

if __name__ == '__main__':
    pass
