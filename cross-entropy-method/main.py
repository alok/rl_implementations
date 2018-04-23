#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch.distributions import Normal

mean, std = 0, 1
# Number of samples to take and size of elite set.
N, n = 100, 10
ITERS = 100

# TODO Apply objective function to X and sort by those indices.

for t in range(ITERS):
    X = Normal(mean, std).sample_n(N)
    X, _ = X.sort(dim=0, descending=False)  # swap descending to min/max objective
    elite = X[:n]
    mean, std = elite.mean(), elite.std()

if __name__ == '__main__':
    print(mean, std)
