#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch.nn import Linear, ReLU, Sequential

from env import A, S

# Q: Callable[[State], List[Action]]
Q = Sequential(
    Linear(S, 128),
    ReLU(),
    Linear(128, 128),
    ReLU(),
    Linear(128, A),
).cuda()
