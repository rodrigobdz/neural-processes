#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch import distributions

# Local imports
from .mlp import MLP


class Encoder(nn.Module):

    # encoder input: x, y
    def __init__(self, in_features, out_features, h_size):
        """NP latent encoder.

        Args:
        in_features: An Int representing the  size of each input sample.
        out_features: An iterable containing the output sizes of the encoding MLP.
        h_size: An Int representing the latent dimensionality.
        """
        super(Encoder, self).__init__()

        self._mlp = MLP(in_features+1, out_features)

        # self._reduce = (out_features[-1] + h_size) // 2
        # shared mapping parameters (for reduction) between mu and sigma layer
        # self._shared_layer = nn.ModuleList([nn.Linear(out_features[-1], self._reduce), nn.ReLU()])
        # self._mu = nn.Sequential(*self._shared_layer, nn.Linear(self._reduce, h_size))
        # self._log_sigma = nn.Sequential(*self._shared_layer, nn.Linear(self._reduce, h_size))

        self._map = nn.Sequential(
            nn.Linear(out_features[-1], h_size), nn.ReLU())
        self._mu = nn.Linear(h_size, h_size)
        self._log_sigma = nn.Linear(h_size, h_size)

        # self._map = nn.Sequential(nn.Linear(out_features[-1], self._reduce), nn.ReLU())
        # self._mu = nn.Linear(self._reduce, h_size)
        # self._log_sigma = nn.Linear(self._reduce, h_size)

    def forward(self, x, y):
        """Encodes the inputs into one representation.

        Args:
        x: Tensor of either shape [B, observations, d_x] (1d regression) or
            [B,observations,d_x2, d_x1] (2d regression).
        y: Tensor of shape [B,observations,d_y].

        Returns:
        A normal distribution over tensors of shape [B, num_latents]
        """

        data = torch.cat((x, y), dim=-1)

        # produce representations r_i and aggregate them (mean)
        # to receive a single order-invariant representation r (crucial for run time redction to
        # O(n+m) (context + target))
        r = self._mlp(data).mean(dim=1) #rows represent r_i
        r = self._map(r)
        # r = nn.ReLU()(r)

        # use representation r to parameterise a MvNormal using a second MLP
        # (_mu and _log_sigma share params except their last layer)
        mu = self._mu(r)
        log_sigma = self._log_sigma(r)
        sigma = .1 + .9 * torch.sigmoid(log_sigma)  # mapping to range .0.:1.

        return distributions.Normal(mu, sigma)
