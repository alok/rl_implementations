#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.nn import Linear

from utils import ParamDict

Weights = ParamDict
Task = DataLoader
criterion = F.mse_loss

CUDA_AVAILABLE = torch.cuda.is_available()

PLOT = True
INPUT_SIZE = 1
OUTPUT_SIZE = 1
LR = 0.02
HIDDEN_SIZE = 64
# BATCH_SIZE = 2**7
BATCH_SIZE = 10
EPOCHS = 1
META_LR_START = 1e-1  # copycat
META_EPOCHS = 30_000
META_BATCH_SIZE = 1

N = 50  # 50 ps on sine wave


def cuda(x):
    return x.cuda() if CUDA_AVAILABLE else x


def gen_task(input_size=INPUT_SIZE) -> Tuple[Task, float, float]:
    # amplitude
    a = np.random.uniform(low=0.1, high=5)  # amplitude
    b = np.random.uniform(low=0, high=2 * np.pi)  # phase

    x = np.linspace(-5, 5, 50)[:, None]
    y = a * np.sin(x + b)

    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).float())

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=CUDA_AVAILABLE,  # TODO change to use cuda if available
    )

    return loader, a, b


class Model(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        I, O = INPUT_SIZE, OUTPUT_SIZE
        H = HIDDEN_SIZE
        self.fc1 = Linear(I, H)
        self.fc2 = Linear(H, H)
        self.out = Linear(H, O)

        # This has to be after the weight initializations or else we get a
        # KeyError.
        if weights is not None:
            self.load_state_dict(deepcopy(weights))

    def forward(self, x):
        r = F.relu

        x = r(self.fc1(x))
        x = r(self.fc2(x))
        x = self.out(x)

        return x


def evaluate(model: Model, task: Task) -> float:
    model.eval()

    # evaluate on all the data at once
    x, y = cuda(Variable(task.dataset.data_tensor)), cuda(Variable(task.dataset.target_tensor))
    preds = model(x)

    loss = criterion(preds, y)
    return float(loss)


def sgd(meta_weights: Weights, k: int) -> Weights:
    task, _, _ = gen_task()

    model = cuda(Model(meta_weights))

    model.train()

    opt = SGD(model.parameters(), lr=LR)

    for i, (x, y) in enumerate(task):
        x, y = cuda(Variable(x)), cuda(Variable(y))
        pred = model(x)
        loss = criterion(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # print(evaluate(model, task))

    return ParamDict(model.state_dict())


if __name__ == '__main__':

    plot_task, a, b = gen_task()
    # need to put model on gpu first for tensors to have the right type
    meta_weights = ParamDict(cuda(Model()).state_dict())
    x_all = cuda(Variable(torch.linspace(-5, 5, 50)[:, None]))
    x_plot = x_all[torch.randperm(len(x_all))[:BATCH_SIZE]]
    y_plot = a * torch.sin(x_plot + b)
    y_all = a * torch.sin(x_all + b)

    for iteration in range(META_EPOCHS):

        weights = [sgd(meta_weights, k=EPOCHS) for _ in range(META_BATCH_SIZE)]
        # the mul by 0 is to get a paramdict of the right size as the start value for summation.
        # TODO implement custom optimizer that makes this work with Adam easily
        META_LR = META_LR_START * (1 - iteration / META_EPOCHS)  # linearly schedule meta-learning rate
        meta_weights = meta_weights + (META_LR / len(weights)) * sum(
            (w - meta_weights for w in weights), 0 * weights[0]
        )

        if (iteration == 0 or (iteration + 1) % 1000) == 0:

            if PLOT:
                plt.cla()

            # meta_weights_before = deepcopy(meta_weights)

            model = cuda(Model(meta_weights))
            model.train()  # set train mode
            opt = SGD(model.parameters(), lr=LR)

            task, a, b = gen_task()

            if PLOT:
                plt.plot(
                    x_all.data.numpy(),
                    model(x_all).data.numpy(),
                    label="pred after 0",
                    color=(0, 0, 1)
                )

            for i, (x, y) in enumerate(task):
                x, y = cuda(Variable(x)), cuda(Variable(y))

                preds = model(x)
                loss = criterion(preds, y)
                print(float(loss))
                opt.zero_grad()
                loss.backward()
                opt.step()

                if (i + 1) % 1 == 0:
                    frac = (i + 1) / len(task)

                    if PLOT:
                        plt.plot(
                            x_all.data.numpy(),
                            model(x_all).data.numpy(),
                            label=f"pred after {i+1}",
                            color=(frac, 0, 1 - frac),
                        )

            if PLOT:
                plt.plot(
                    x_all.data.numpy(),
                    y_all.data.numpy(),
                    label="true",
                    color=(0, 1, 0),
                )
                plt.plot(
                    x_plot.data.numpy(),
                    y_plot.data.numpy(),
                    "x",
                    label="train",
                    color="k",
                )

            if PLOT:
                plt.ylim(-4, 4)
                plt.legend(loc="lower right")
                plt.pause(0.01)

            # test model
            print(evaluate(model, plot_task))

            # model.load_state_dict(meta_weights_before)  # restore from snapshot

            print(72 * '-')
            print(f"iteration               {iteration+1}")
            print(f"loss on plotted curve   {float(loss):.3f}")
