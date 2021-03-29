#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch import optim
from torch import distributions

# Local imports

from .plot import plot_1d as _plot_1d
from .encoder import Encoder
from .decoder import Decoder


class NeuralProcess(nn.Module):

    def __init__(self, in_features, encoder_out, decoder_out, h_size, mc_size):
        super(NeuralProcess, self).__init__()
        # self._in_features = in_features
        # self._encoder_out = encoder_out
        # self._decoder_out = decoder_out
        # self._h_size = h_size

        self._mc_size = mc_size
        self._encoder = Encoder(in_features, encoder_out, h_size)
        self._decoder = Decoder(in_features, decoder_out, h_size)

    def forward(self, context_x, context_y, target_x, target_y=None):

        # q_prior will alyways be defined
        q_prior = self._encoder(context_x, context_y)

        # train time behaviour
        if target_y is not None:
            q_posterior = self._encoder(target_x, target_y)
            # rsample() takes care of rep. trick (z = µ + σ * I * ϵ , ϵ ~ N(0,1))
            z = q_posterior.rsample()

            # monte carlo sampling for integral over logp
            # z will be concatenate to every x_i and therefore must match
            # dimensionality of x_i
            # z = q_posterior.rsample([self._mc_size])
            # z = z[:, :, None, :].expand(-1, -1, target_x.shape[1], -1)
            # z = z.permute(1, 0, 2, 3)
            # target_x = target_x[:, None, :,
            #                     :].expand(-1, self._mc_size, -1, -1)

        # test time behaviour
        else:
            # rsample() takes care of rep. trick (z = µ + σ * I * ϵ , ϵ ~ N(0,1))
            z = q_prior.rsample()

        z = z[:, None, :].expand(-1, target_x.shape[1], -1)

        mu, sigma, distr = self._decoder(target_x, z)

        train = target_y is not None  # true at train time
        q = (q_prior, q_posterior) if train else q_prior

        return (mu, sigma, distr), q

    def fit(self, niter, save_iter, train_set, query_test, learning_rate=1e-4):

        opt = optim.Adam(self.parameters(), lr=learning_rate)

        for i in range(niter):
            self.train()
            context_x, context_y, target_x, target_y = train_set[i]
            distr_tuple, q = self(context_x, context_y, target_x, target_y)

            predict_distr = distr_tuple[2]
            prior = q[0]
            posterior = q[1]

            training_loss = NeuralProcess.loss(predict_distr, target_y,
                                               prior, posterior, self._mc_size)
            training_loss.backward()
            opt.step()
            opt.zero_grad()

            if i % save_iter == 0:
                self.eval()
                with torch.no_grad():
                    # (mu, sigma, _), _ = self(context_x, context_y, target_x)
                    # plot_functions(target_x, target_y, context_x, context_y, mu, sigma)

                    # No target_y available at test time
                    context_x, context_y, target_x, target_y = query_test[0]
                    (mu, sigma, predict_distr), q = self(
                        context_x, context_y, target_x)

                    print(f'Iteration: {i}, loss: {training_loss}')
                    _plot_1d(context_x.cpu(), context_y.cpu(), target_x.cpu(),
                             target_y.cpu(), mu.cpu(), sigma.cpu())

        return mu, sigma

    def loss(distr, target_y, prior, posterior, mc_size):

        target_y = target_y[:, None, :, :].expand(-1, mc_size, -1, -1)
        logp = distr.log_prob(target_y).sum(
            dim=2, keepdims=True).mean(dim=1).squeeze()

        # analytic solution exists since two MvGaussians are used
        kl = distributions.kl_divergence(posterior, prior)

        # optimiser uses gradient descent but
        # ELBO should be maximized: therefore -loss
        loss = -torch.mean(logp - kl)

        return loss
