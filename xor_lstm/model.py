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

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from pudb import set_trace
from torch import Tensor, nn
from torch.autograd import Variable
from torch.nn import Parameter, Sequential
from torch.utils.data import DataLoader, Dataset

rnn = nn.LSTM(
    input_size=1,
    hidden_size=5,
    num_layers=3,
)

'00000000010000000000001000000000000000000000000000' '-> 1'


if __name__ == '__main__':
    t = Variable(torch.randn(,2))
