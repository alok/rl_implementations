#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
from collections import OrderedDict
from numbers import Number
from typing import NamedTuple

import numpy as np
import torch
from torch.autograd import Variable


class Step(NamedTuple):
    """One step of a rollout."""
    state: Variable
    action: Variable
    succ_state: Variable
    reward: float
    done: bool


def np_to_var(arr: np.ndarray) -> Variable:
    return Variable(torch.from_numpy(arr).float())


class ReplayBuffer(list):
    def __init__(self, buffer_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer_size = buffer_size

    def append(self, transition):

        if len(self) < self.buffer_size:
            super().append(transition)
        else:
            idx = len(self) % self.buffer_size
            self[idx] = transition

    def sample(self, batch_size):
        return random.sample(self, batch_size)


class ParamDict(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def __add__(self, other):
        if isinstance(other, dict):
            assert other.keys() == self.keys()
            return ParamDict({k: self[k] + other[k] for k in self})
        else:
            return NotImplementedError

    def __rmul__(self, other):
        if isinstance(other, Number):
            return ParamDict({k: other * v for k, v in self.items()})
        else:
            return NotImplementedError

    __mul__ = __rmul__
