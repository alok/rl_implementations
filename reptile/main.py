#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy

import numpy as np
import ray
import torch
import torch.nn.functional as F
from torch import Tensor, linspace, nn, randperm, sin
from torch.autograd import Variable
from torch.nn import Linear
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from utils import ParamDict as P

# To avoid tkinter not installed error on headless server
try:
    import matplotlib
    matplotlib.use('AGG')
    import matplotlib.pyplot as plt
except:
    pass

Weights = P
criterion = F.mse_loss

CUDA_AVAILABLE = torch.cuda.is_available()

PLOT = True
INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE = 1, 64, 1
N = 50  # Use 50 evenly spaced points on sine wave.

LR, META_LR = 0.02, 0.1  # Copy OpenAI's hyperparameters.
BATCH_SIZE, META_BATCH_SIZE = 10, 3
EPOCHS, META_EPOCHS = 1, 30_000
TEST_GRAD_STEPS = 2**3
PLOT_EVERY = 3_000


def cuda(x):
    return x.cuda() if CUDA_AVAILABLE else x


def shuffle(*tensors, length=None):
    """Shuffle multiple tensors in the same way.

    All tensors must have the same length in the first dimension.
    """

    perm = randperm(len(tensors[0]))[:length]

    return [tensor[perm] for tensor in tensors]


def gen_task(num_pts=N) -> DataLoader:
    # amplitude
    a = np.random.uniform(low=0.1, high=5)  # amplitude
    b = np.random.uniform(low=0, high=2 * np.pi)  # phase

    # Need to make x N,1 instead of N, to avoid
    # https://discuss.pytorch.org/t/dataloader-gives-double-instead-of-float
    x = linspace(-5, 5, num_pts)[:, None].float()
    y = a * sin(x + b).float()

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

    x, y = cuda(Variable(task.dataset.data_tensor)), cuda(
            Variable(task.dataset.target_tensor)
            )
    loss = criterion(model(x), y)
    return float(loss)


@ray.remote
def sgd(meta_weights: Weights, epochs: int) -> Weights:
    """Run SGD on a randomly generated task."""

    model = cuda(Model(meta_weights))
    model.train()  # Ensure model is in train mode.

    task = gen_task()
    opt = SGD(model.parameters(), lr=LR)

    for epoch in range(epochs):
        for x, y in task:
            train_batch(x, y, model, opt)

    return model.state_dict()


def REPTILE(
        meta_weights: Weights,
        meta_batch_size: int = META_BATCH_SIZE,
        epochs: int = EPOCHS,
        ) -> Weights:
    """Run one iteration of REPTILE."""
    weights = ray.get([
        sgd.remote(meta_weights, epochs) for _ in range(meta_batch_size)
        ])
    weights = [P(w) for w in weights]

    # TODO Implement custom optimizer that makes this work with builtin
    # optimizers easily. The multiplication by 0 is to get a ParamDict of the
    # right size as the identity element for summation.
    meta_weights += (META_LR / epochs) * sum((w - meta_weights
        for w in weights), 0 * meta_weights)
    return meta_weights


if __name__ == '__main__':
    try:
        ray.init()
    except Exception as e:
        print(e)

    # Need to put model on GPU first for tensors to have the right type.
    meta_weights = cuda(Model()).state_dict()

    if PLOT:
        # Generate fixed task to evaluate on.
        plot_task = gen_task()

        x_all, y_all = plot_task.dataset.data_tensor, plot_task.dataset.target_tensor
        x_plot, y_plot = shuffle(x_all, y_all, length=10)

        # Set up plot
        fig, ax = plt.subplots()
        true_curve = ax.plot(
                x_all.numpy(),
                y_all.numpy(),
                label='True',
                color='g',
                )

        ax.plot(
                x_plot.numpy(),
                y_plot.numpy(),
                'x',
                label='Training points',
                color='k',
                )

        ax.legend(loc="lower right")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)

    for iteration in range(1, META_EPOCHS + 1):

        meta_weights = REPTILE(P(meta_weights))

        if iteration == 1 or iteration % PLOT_EVERY == 0:

            model = cuda(Model(meta_weights))
            model.train()  # set train mode
            opt = SGD(model.parameters(), lr=LR)

            for _ in range(TEST_GRAD_STEPS):
                train_batch(x_plot, y_plot, model, opt)

            if PLOT:

                ax.set_title(f'REPTILE after {iteration:n} iterations.')
                curve, = ax.plot(
                        x_all.numpy(),
                        model(Variable(x_all)).data.numpy(),
                        label=f'Pred after {TEST_GRAD_STEPS:n} gradient steps.',
                        color='r',
                        )

                plt.savefig(f'figs/{iteration}.png')

                ax.lines.remove(curve)

            print(f'Iteration: {iteration}\tLoss: {evaluate(model, plot_task):.3f}')

