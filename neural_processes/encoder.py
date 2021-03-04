#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as _torch
from torch import nn as _nn
from torch import distributions as _distributions

# Local imports
from .mlp import MLP


class Encoder(_nn.Module):

    # encoder input: x, y
    def __init__(self, in_features, out_features, h_size):
        """NP latent encoder.

        Args:
        in_features: An Int representing the  size of each input sample.
        out_features: An iterable containing the output sizes of the encoding MLP.
        h_size: An Int representing the latent dimensionality.
        """
        super(Encoder, self).__init__()
        # self._in_features = in_features + 1 # adding one dim since x and y will be concatenated
        # self._h_size = h_size
        # self._out_features = out_features
        self._mlp = MLP(in_features+1, out_features)

        # self._reduce = (out_features[-1] + h_size) // 2
        # shared mapping parameters (for expansion) between mu and sigma layer
        # self._shared_layer = _nn.ModuleList([_nn.Linear(out_features[-1], self._reduce), _nn.ReLU()])
        # self._mu = _nn.Sequential(*self._shared_layer, _nn.Linear(self._reduce, h_size))
        # self._log_sigma = _nn.Sequential(*self._shared_layer, _nn.Linear(self._reduce, h_size))
        # self._mu = _nn.Linear(out_features[-1], h_size)
        # self._log_sigma = _nn.Linear(out_features[-1], h_size)

        self._map = _nn.Sequential(_nn.Linear(out_features[-1], h_size), _nn.ReLU())
        self._mu = _nn.Linear(h_size, h_size)
        self._log_sigma = _nn.Linear(h_size, h_size)


    def forward(self, x, y):
        """Encodes the inputs into one representation.

        Args:
        x: Tensor of either shape [B, observations, d_x].
        y: Tensor of shape [B,observations,d_y].

        Returns:
        A Multivariate Gaussian (independence assumed) over tensors of shape [B, num_latents]
        """


        data = _torch.cat((x, y), dim=-1)

        # produce representations r_i and aggregate them (mean)
        # to receive a single order-invariant representation r (crucial for run time redction to
        # O(n+m) (context + target))
        r = self._mlp(data).mean(dim=1)  # rows represent r_i
        r = self._map(r)

        # use representation r to parameterise a MvNormal using a second MLP
        # (_mu and _log_sigma share params except their last layer)
        mu = self._mu(r)

        # mapping to range .0.:1. (from colab)
        sigma = .1 + .9 * _torch.sigmoid(self._log_sigma(r))
        sigma = sigma.diag_embed() #lower factor of Cholesky

        return _distributions.MultivariateNormal(mu, scale_tril=sigma)
