#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch import optim
from torch import distributions

# Local imports

from .plot import plot_2d
from .model import NeuralProcess

from .utils import preprocess_mnist


class ImgNeuralProcess(nn.Module):

    def __init__(self, in_features, encoder_out, decoder_out, h_size):
        super(ImgNeuralProcess, self).__init__()

        self._np = NeuralProcess(
            in_features, encoder_out, decoder_out, h_size)

    def forward(self, context_x, context_y, target_x, target_y=None):

        if target_y is not None:
            self._np(context_x, context_y, target_x, target_y)
        else:
            self._np(context_x, context_y, target_y)

    def _fit(self, epochs, save_epoch, train_generator, test_generator, opt):

        running_loss = 0.0
        losses = []
        nll = []
        kll = []

        for j in range(epochs):
            for i, (Y, _) in enumerate(train_generator):
                train_set = preprocess_mnist(Y, train=True)

                context_x, context_y, target_x, target_y = train_set
                context_x, context_y, target_x, target_y = train_set[i]
                distr_tuple, q = self(context_x, context_y, target_x, target_y)

                predict_distr = distr_tuple[2]
                prior, posterior = q

                loss = self._np._loss(
                    predict_distr, target_y, prior, posterior, nll, kll)
                running_loss += loss.item()

                loss.backward()
                opt.step()
                opt.zero_grad()

                if i % 1000 == 0:
                    with torch.no_grad():
                        print(f'Epoch: {j}, Iteration: {i}, loss: {loss}')

            if j % save_epoch == 0:
                # np.eval()
                with torch.no_grad():
                    Y, label = next(iter(test_generator))
                    query_test = preprocess_mnist(Y, train=False)

                    context_x, context_y, target_x, target_y = query_test
                    (mu, sigma, predict_distr), q = self._np(
                        context_x, context_y, target_x)

                    print(f'Iteration: {i}, loss: {loss}')
                    plot_2d(context_x, context_y,
                            target_x, mu, target_y, label)

        return mu, sigma, (losses, nll, kll)
