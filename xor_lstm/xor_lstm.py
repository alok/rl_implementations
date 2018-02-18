#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import operator
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor, nn
from torch.autograd import Variable
from torch.nn import Linear, ReLU, Softmax

CUDA_AVAILABLE = torch.cuda.is_available()

N = NUM_SAMPLES = 100_000
D = INPUT_DIM = 1
C = NUM_CLASSES = 2
MAX_SEQ_LEN = 50
BATCH_SIZE = 1

OP = operator.xor


def foldr(arr: np.ndarray, op) -> np.ndarray:
    """Specific version of foldr that's only for Numpy arrays"""

    return np.fromiter(itertools.accumulate(arr, op), dtype=np.float32, count=len(arr))


# TODO generate minibatch in one shot


def gen_data(seq_len=MAX_SEQ_LEN, batch_size=BATCH_SIZE, input_dim=D):
    # We want binary data here.
    xs = np.random.randint(low=0, high=2, size=(seq_len, batch_size, input_dim))

    # Have to do some weird slicing here to do the fold in one shot.

    # Need to stack along last axis or else we are going to have to transpose
    # the result to get the labels in the right order when flattened.
    ys = np.stack([foldr(xs[:, i, 0], OP) for i in range(xs.shape[1])], axis=-1)

    xs = Variable(torch.from_numpy(xs).float())  # Binary data needs to be cast to float.
    ys = Variable(torch.from_numpy(ys).long())

    return (xs.cuda(), ys.cuda()) if CUDA_AVAILABLE else (xs, ys)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # Make hidden dimension large enough to avoid XOR's local minima, an
        # issue only really found in such small problems.
        H = 20

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=H,
            num_layers=2,
        )
        self.fc = Linear(H, C)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


def argmax(tensor, dim=1):
    return tensor.max(dim=dim)[1]


# Hack to check if we've already trained a model (assumed to be a good one.
model_path = Path(f'model-{OP.__name__}' + ('cuda' if CUDA_AVAILABLE else '') + '.pth')

test_mode = model_path.exists()
train_mode = not test_mode

if test_mode:
    model = torch.load(model_path)
    # Need this to avoid non_contiguous memory warning.
    model.lstm.flatten_parameters()
    model.eval()
else:
    model = Model().cuda() if CUDA_AVAILABLE else Model()
    model.train()

if __name__ == '__main__':

    criterion = F.cross_entropy
    optimizer = optim.Adam(model.parameters())

    for i in range(N // BATCH_SIZE):

        # TODO pack padded sequence
        seq_len = np.random.randint(low=1, high=MAX_SEQ_LEN + 1)
        batch_size = BATCH_SIZE

        xs, ys = gen_data(
            seq_len=seq_len,
            batch_size=batch_size,
        )

        predictions = model(xs)

        loss = criterion(predictions.view(-1, C), ys.view(-1))

        if i % (N // 200) == 0:
            print(
                f'Iteration: {i}',
                f'Length: {seq_len}',
                f'Loss: {float(loss)}',
                sep=' | ',
            )

        # Stop early.
        if train_mode and float(loss) < 1e-6:
            print(72 * '-')
            print(
                f'Iteration: {i}',
                f'Length: {seq_len}',
                f'Loss: {float(loss)}',
                sep=' | ',
            )

            break

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if train_mode:
        torch.save(model, model_path)
