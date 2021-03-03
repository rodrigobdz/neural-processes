#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adapted from:
    https://github.com/deepmind/neural-processes/blob/aca7e38ea64b718fbd7f311ccae5d51d73447d15/attentive_neural_process.ipynb

Copyright 2019 Google LLC
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0
"""

import torch as _torch


class GPCurves:
    """Generates curves using a Gaussian Process (GP).

    Supports vector inputs (x) and vector outputs (y). A (scaled) RBF Kernel is used to
    output independent Gaussian Processes.
    (*) Scale parameters sigma and length_scale can be passed to define a specific RBF Kernel
    otherwise the parameters will be sampled uniformly within [0., length_scale) and [0., sigma_scale)
    """

    def __init__(self,
                batch_size,
                max_num_context,
                x_size=1,
                y_size=1,
                length_scale=1.0,
                sigma_scale=1.0,
                random_params=True,
                testing=False,
                dev='cpu'
                ):

        """Creates a regression dataset of functions sampled from a GP.

        Args:
            batch_size: An integer.
            max_num_context: The max number of observations in the context.
            x_size: Integer >= 1 for length of "x values" vector.
            y_size: Integer >= 1 for length of "y values" vector.
            length_scale: Float; typical scale for kernel distance function.
            sigma_scale: Float; typical scale for variance.
            random_params: If `True`, the kernel parameters (length and sigma)
            will be sampled uniformly within [0.1, length_scale] and [0.1, sigma_scale].
            dev: Either `cpu` or `gpu`, tensors will be casted to appropriate device
        """

        self._batch_size = batch_size
        self._max_num_context = max_num_context
        self._x_size = x_size
        self._y_size = y_size
        self._length_scale = length_scale
        self._sigma_scale = sigma_scale
        self._random_params = random_params
        self._testing = testing
        self._dev = dev

    def _kernel(self, X, length, sigma, noise=2e-2):
        """Returns a (scaled) RBF kernel used to init the GP

        Args:
            X: Tensor of shape [B, num_total_points, x_size] with
                the values of the x-axis data.
            length: Tensor of shape [B, y_size, x_size], the scale
                parameter of the Gaussian kernel.
            sigma: Tensor of shape [B, y_size], the magnitude
                of the std.
            sigma_noise: Float, std of the noise that we add for stability.

            Returns:
            The kernel, a float tensor of shape
            [B, y_size, num_total_points, num_total_points].

        """
        num_total_points = X.size()[1]

        # X.size(): [B, num_total_points, x_size]
        x1 = X.unsqueeze(1)# [B, 1, num_total_points, x_size]
        x2 = X.unsqueeze(2)# [B, num_total_points, 1, x_size]
        diff = x1 - x2 # [B, num_total_points, num_total_points, x_size]

        # [B, y_size, num_total_points, num_total_points, x_size]
        # None indexing [None, :] acts like tensor.unsqueeze(dim)
        norm = _torch.square(diff[:, None, :, :, :] / length[:, :, None, None, :])

        # [B, data_size, num_total_points, num_total_points]
        norm = norm.sum(-1) # one data point per row

        # [B, y_size, num_total_points, num_total_points]
        kernel = _torch.square(sigma)[:, :, None, None] * _torch.exp(-0.5 * norm)

        # Add some noise to the diagonal to make the cholesky work
        kernel.add_(_torch.eye(num_total_points).mul(noise**2))

        return kernel

    def generate_curves(self):
        """Builds the op delivering the data.

        Generated functions are `float32` with x values between -2 and 2.
        Returns:
        GP data, four float tensors of shape
        [B, num_total_points, dim_x/dim_y]

        """

        num_context = _torch.randint(3, self._max_num_context, [])

        if self._testing:
            num_target = 400
            num_total_points = num_target
            X = _torch.arange(-2, 2, 0.01).unsqueeze(0).expand(self._batch_size, -1)
            # attention! returns view - copy necessary if in place operations are used
            X.unsqueeze_(-1)

        else:
            num_target = _torch.randint(0, self._max_num_context - num_context, [])
            num_total_points = num_context + num_target
            X = _torch.Tensor(self._batch_size, num_total_points, self._x_size).uniform_(-2, 2)


        # set Kernel parameters randomly for every batch
        if self._random_params:
            length = _torch.Tensor(self._batch_size, self._y_size, self._x_size).uniform_(0.1, self._length_scale)
            sigma = _torch.Tensor(self._batch_size, self._y_size).uniform_(0.1, self._sigma_scale)

        else:
        # use the same Kernel parameters for every batch
            length = _torch.ones(self._batch_size, self._y_size, self._x_size).mul_(self._length_scale)
            sigma = _torch.ones(self._batch_size, self._y_size).mul_(self._sigma_scale)


        # [batch_size, y_size, num_total_points, num_total_points]
        kernel = self._kernel(X, length, sigma)

        # change precision to float64 for Cholesky and cast to float32 afterwards
        cholesky = kernel.double().cholesky().float()

        # sampling with no mean assumption: y = mu + sigma*z~N(0,I) ~ c.L * rand_normal([0, 1]) with appropriate shape
        y = cholesky.matmul(_torch.randn(self._batch_size, self._y_size, num_total_points, 1))

        # [batch_size, num_total_points, y_size]
        Y = y.squeeze(3).permute(0, 2, 1)

        if self._testing:
            # Select the targets
            target_x = X
            target_y = Y

            # Select the observations
            idx = _torch.randperm(num_target)
            context_x = _torch.index_select(X, 1, idx[:num_context])
            context_y = _torch.index_select(Y, 1, idx[:num_context])
        else:
            # Select the targets which will consist of the context points as well as
            # some new target points
            target_x = X[:, :num_target + num_context, :]
            target_y = Y[:, :num_target + num_context, :]

            # Select the observations
            context_x = X[:, :num_context, :]
            context_y = Y[:, :num_context, :]

        context_x = context_x.to(self._dev)
        context_y = context_y.to(self._dev)
        target_x = target_x.to(self._dev)
        target_y = target_y.to(self._dev)


        return (context_x, context_y, target_x, target_y)
