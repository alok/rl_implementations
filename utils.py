#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import operator
import random
from collections import OrderedDict
from copy import deepcopy
from numbers import Number
from typing import NamedTuple

import numpy as np
import torch
from scipy.signal import lfilter
from torch import Tensor
from torch.autograd import Variable


class Step(NamedTuple):
    """One step of a rollout."""

    state: Variable
    action: Variable
    reward: float
    succ_state: Variable
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

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            return NotImplemented

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)


def G(x, discount):
    if isinstance(x, (Tensor, Variable)):
        assert x.ndimension >= 1
        return lfilter([1], [1, -discount], x.numpy()[::-1], axis=0)[::-1]
    elif isinstance(x, np.ndarray):
        assert x.ndim >= 1
    return lfilter([1], [1, -discount], x[::-1], axis=0)[::-1]
