#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as _torch
from torch import nn as _nn
from torch import distributions as _distributions

# Local imports
from .mlp import MLP


class Decoder(_nn.Module):

    def __init__(self, in_features_x, out_features, h_size):
        super(Decoder, self).__init__()
        # self._in_features_x = in_features_x
        # self._h_size = h_size
        # self._in_features = in_features_x + h_size
        # self._out_features = out_features # should end with [...,2]

        # make use of amortized variational inference and only
        # (no parameter fitting per data point instead fit a function)
        # learn one function that outputs distribution parameters
        self._mlp = MLP(in_features_x+h_size, out_features)

    def forward(self, z, target):

        data = _torch.cat((z, target), dim=-1)

        mu, log_sigma = self._mlp(data).split(1, dim=-1)

        sigma = .1 + .9 * _nn.Softplus()(log_sigma) # bound variance to range 0:1
        # ANP paper: use Softplus instead of Sigmoid, .1 + ... for Cholesky


        return mu, sigma, _distributions.Normal(mu, sigma)
