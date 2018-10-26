#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random


class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def append(self, transition):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition)
        else:
            idx = len(self.buffer) % self.buffer_size
            self.buffer[idx] = transition

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


if __name__ == "__main__":
    pass
