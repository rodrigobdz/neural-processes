#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multilayer perceptron
"""

from torch import nn as _nn


class MLP(_nn.Module):

    def __init__(self, in_features, out_features):

        super(MLP, self).__init__()

        # 1st layer of the MLP
        # ReLU used in next step
        self.linears = _nn.ModuleList(
            [_nn.Linear(in_features, out_features[0])]
        )

        # ouput_sizes generically
        # thus generic MLP creation necessary
        for i, size in enumerate(out_features[:-1]):
            self.linears.extend(_nn.ModuleList(
                [_nn.ReLU(),
                _nn.Linear(size, out_features[i+1])
                ]
                )
            )

    def forward(self, x):

        for linear in self.linears:
            x = linear(x)

        return x
