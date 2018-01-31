#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# CartPole so discrete actions

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

import torch
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch import nn
from torch.autograd import Variable
from torch.nn import Linear, Parameter, ReLU, Sequential, Softmax
from torch.utils.data import DataLoader, Dataset


class Model(nn.Module):
    def __init__(self, env):

        S, A = int(np.prod(env.observation_space.shape)), int(env.action_space.n)
        H = 50
        self.model = nn.Sequential(
            Linear(S, H),
            ReLU(),
            Linear(H, H),
            ReLU(),
            Linear(H, H),
            ReLU(),
            Softmax(H, A),
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    pass
