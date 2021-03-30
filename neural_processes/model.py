#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch import optim
from torch import distributions

# Local imports

from .plot import plot_1d
from .encoder import Encoder
from .decoder import Decoder


class NeuralProcess(nn.Module):

    def __init__(self, in_features, encoder_out, decoder_out, h_size, opt):
        super(NeuralProcess, self).__init__()

        self._encoder = Encoder(in_features, encoder_out, h_size)
        self._decoder = Decoder(in_features, decoder_out, h_size)
        self._opt = opt

    def forward(self, context_x, context_y, target_x, target_y=None):

        # q_prior will alyways be defined
        q_prior = self._encoder(context_x, context_y)

        # train time behaviour
        if target_y is not None:
            q_posterior = self._encoder(target_x, target_y)
            # one sample MC estimate
            # rsample() takes care of rep. trick (z = µ + σ * I * ϵ , ϵ ~ N(0,1))
            z = q_posterior.rsample()

        # test time behaviour
        else:
            # rsample() takes care of rep. trick (z = µ + σ * I * ϵ , ϵ ~ N(0,1))
            z = q_prior.rsample()

        z = z[:, None, :].expand(-1, target_x.shape[1], -1)

        mu, sigma, distr = self._decoder(target_x, z)

        train = target_y is not None  # true at train time
        q = (q_prior, q_posterior) if train else q_prior

        return (mu, sigma, distr), q

    def _fit(self, niter, save_iter, train_set, query_test):

        running_loss = 0.0
        losses = []
        nll = []
        kll = []

        for i in range(niter):
            self.train()
            context_x, context_y, target_x, target_y = train_set[i]
            distr_tuple, q = self(context_x, context_y, target_x, target_y)

            predict_distr = distr_tuple[2]
            prior, posterior = q


            loss = NeuralProcess._loss(predict_distr, target_y, prior, posterior, nll, kll)

            running_loss += loss.item()

            loss.backward()
            self._opt.step()
            self._opt.zero_grad()

            if i % save_iter == 0:
                self.eval()
                losses.append(running_loss/save_iter)
                running_loss = 0.0

                with torch.no_grad():

                    # No target_y available at test time
                    ind = torch.randint(0, len(query_test), [])
                    context_x, context_y, target_x, target_y = query_test[ind]
                    (mu, sigma, predict_distr), q = self(
                        context_x, context_y, target_x)

                    print(f'Iteration: {i}, loss: {loss}')
                    plot_1d(context_x.cpu(), context_y.cpu(), target_x.cpu(),
                            target_y.cpu(), mu.cpu(), sigma.cpu())

        return mu, sigma, (losses, nll, kll)

    def _loss(self, predict_distr, target_y, prior, posterior, nll, kll):

        logp = predict_distr.log_prob(target_y).squeeze()

        # analytic solution exists since two MvGaussians are used
        # kl of shape [batch_size]
        kl = distributions.kl_divergence(
            posterior, prior).sum(dim=1)  # [batch_size]
        # [batch_size, num_points]
        kl = kl[:, None].expand(-1, target_y.shape[1])
        kl = kl / target_y.shape[1]  # [batch_size, num_points]

        # optimiser uses gradient descent but
        # ELBO should be maximized: therefore -loss

        nll.append(-logp.mean().detach().item())
        kll.append(kl.mean().detach().item())

        # mini batch gradient
        loss = -torch.mean(logp - kl)

        return loss
