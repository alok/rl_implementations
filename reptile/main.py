#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Variable
from torch.nn import Linear
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from utils import ParamDict as P

Weights = P
criterion = F.mse_loss

CUDA_AVAILABLE = torch.cuda.is_available()

PLOT = True
INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE = 1, 64, 1
N = 50  # Use 50 evenly spaced points on sine wave.

LR, META_LR = 0.02, 0.1  # Copy OpenAI's hyperparameters.
BATCH_SIZE, META_BATCH_SIZE = 10, 1
EPOCHS, META_EPOCHS = 1, 30_000
TEST_GRAD_STEPS = 1  # A single gradient step can work well.


def cuda(x):
    return x.cuda() if CUDA_AVAILABLE else x


def gen_task(num_pts=N) -> DataLoader:
    # amplitude
    a = np.random.uniform(low=0.1, high=5)  # amplitude
    b = np.random.uniform(low=0, high=2 * np.pi)  # phase

    # Need to make x N,1 instead of N, to avoid
    # https://discuss.pytorch.org/t/dataloader-gives-double-instead-of-float
    x = torch.linspace(-5, 5, num_pts)[:, None].float()
    y = a * torch.sin(x + b).float()

    dataset = TensorDataset(x, y)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=CUDA_AVAILABLE,
    )

    return loader


class Model(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.fc1 = Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.fc2 = Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.out = Linear(HIDDEN_SIZE, OUTPUT_SIZE)

        # This has to be after the weight initializations or else we get a
        # KeyError.
        if weights is not None:
            self.load_state_dict(deepcopy(weights))

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


def train_batch(x: Tensor, y: Tensor, model: Model, opt) -> None:
    """Statefully train model on single batch."""
    x, y = cuda(Variable(x)), cuda(Variable(y))

    loss = criterion(model(x), y)

    opt.zero_grad()
    loss.backward()
    opt.step()


def evaluate(model: Model, task: DataLoader, criterion=criterion) -> float:
    """Evaluate model on all the task data at once."""
    model.eval()

    x, y = cuda(Variable(task.dataset.data_tensor)), cuda(Variable(task.dataset.target_tensor))
    loss = criterion(model(x), y)
    return float(loss)


def sgd(meta_weights: Weights, epochs: int) -> Weights:
    """Run SGD on a randomly generated task."""

    model = cuda(Model(meta_weights))
    model.train()  # ensure model is in train mode

    task = gen_task()
    opt = SGD(model.parameters(), lr=LR)

    for epoch in range(epochs):
        for i, (x, y) in enumerate(task):
            train_batch(x, y, model, opt)

    return P(model.state_dict())


def REPTILE(meta_weights: Weights, meta_batch_size=META_BATCH_SIZE, epochs=EPOCHS) -> Weights:
    """Run one iteration of REPTILE."""
    weights = [sgd(meta_weights, epochs) for _ in range(meta_batch_size)]

    # TODO Implement custom optimizer that makes this work with builtin
    # optimizers easily. The multiplication by 0 is to get a ParamDict of the
    # right size as the identity element for summation.
    meta_weights += (META_LR / len(weights)) * sum((w - meta_weights
                                                    for w in weights), 0 * weights[0])
    return meta_weights


if __name__ == '__main__':

    plot_task = gen_task()  # Generate fixed task to evaluate on.
    # Need to put model on GPU first for tensors to have the right type.
    meta_weights = P(cuda(Model()).state_dict())

    x_all, y_all = plot_task.dataset.data_tensor, plot_task.dataset.target_tensor

    for iteration in range(META_EPOCHS):

        meta_weights = REPTILE(meta_weights)

        if iteration == 0 or (iteration + 1) % 1000 == 0:

            model = cuda(Model(meta_weights))
            model.train()  # set train mode
            opt = SGD(model.parameters(), lr=LR)

            for i in range(TEST_GRAD_STEPS):
                # Training on all the points rather than just a sample works better.
                train_batch(x_all, y_all, model, opt)

            if PLOT:
                plt.cla()

                plt.plot(
                    x_all.numpy(),
                    model(Variable(x_all)).data.numpy(),
                    label=f"Pred after {i+1} gradient steps.",
                    color=(1, 0, 0),
                )

                plt.plot(
                    x_all.numpy(),
                    y_all.numpy(),
                    label="True",
                    color=(0, 1, 0),
                )

                plt.ylim(-4, 4)
                plt.legend(loc="lower right")
                plt.pause(0.01)

            print(f'Iteration: {iteration+1}\tLoss: {evaluate(model, plot_task):.3f}')
