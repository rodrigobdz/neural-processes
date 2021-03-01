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

        # self._expand = (out_features[-1] + h_size) // 2
        # shared mapping parameters (for expansion) between mu and sigma layer
        # self._shared_layer = _nn.ModuleList([_nn.Linear(out_features[-1], self._expand), _nn.ReLU()])
        # self._mu = _nn.Sequential(*self._shared_layer, _nn.Linear(self._expand, h_size))
        # self._log_sigma = _nn.Sequential(*self._shared_layer, _nn.Linear(self._expand, h_size))
        self._mu = _nn.Linear(out_features[-1], h_size)
        self._log_sigma = _nn.Linear(out_features[-1], h_size)

    def forward(self, x, y):
        """Encodes the inputs into one representation.

        Args:
        x: Tensor of either shape [B, observations, d_x].
        y: Tensor of shape [B,observations,d_y].

        Returns:
        A Multivariate Gaussian (independence assumed) over tensors of shape [B, num_latents]
        """

        # TODO: relevant when creating the model:
        #   maybe use DataLoader for training
        # TODO: change design?
        data = _torch.cat((x, y), dim=-1)

        # produce representations r_i and aggregate them (mean)
        # to receive a single order-invariant representation r (crucial for run time redction to
        # O(n+m) (context + target))
        r = self._mlp(data).mean(dim=1) #rows represent r_i
        r = _nn.ReLU()(r)

        # use representation r to parameterise a MvNormal using a second MLP
        # (_mu and _log_sigma share params except their last layer)
        mu = self._mu(r)
        sigma = .1 + .9 * _nn.Sigmoid()(self._log_sigma(r)) # mapping to range .0.:1. (from colab)
        sigma = sigma.diag_embed().tril()

        return _distributions.MultivariateNormal(mu, sigma)
