#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    '-b',
    '--batch_size',
    type=int,
    default=128,
)

parser.add_argument(
    '-r',
    '--replay_buffer_size',
    type=int,
    default=10**4,
)


parser.add_argument(
    '-i',
    '--iterations',
    type=int,
    default=10**3,
)

parser.add_argument(
    '-d',
    '--discount_rate',
    '--gamma',
    type=float,
    default=0.999,
)

parser.add_argument(
    '-e',
    '--exploration_rate',
    '--epsilon',
    type=float,
    default=0.9,
)

parser.add_argument(
    '-l',
    '--lr',
    '--learning_rate',
    type=float,
    default=1e-7,
)

args = parser.parse_args()

if __name__ == '__main__':
    pass
