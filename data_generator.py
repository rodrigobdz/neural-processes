#%%
import torch
#%%

# Adapted from: 
# https://colab.research.google.com/github/deepmind/neural-processes/blob/master/attentive_neural_process.ipynb#scrollTo=Px-atGEfNnWT
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0

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
                sigma_scale=1.0,
                length_scale=1.0,
                random_params=True
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
    """
    self.batch_size = batch_size
    self.max_num_context = max_num_context
    self.x_size = x_size
    self.y_size = y_size
    self.length_scale = length_scale
    self.sigma_scale = sigma_scale
    self.random_params = random_params

    def _kernel(self, X, l1, sigma, noise=1e-4):
        """Returns a (scaled) RBF kernel used to init the GP

        Args:
            X: Tensor of shape [B, num_total_points, x_size] with
                the values of the x-axis data.
            l1: Tensor of shape [B, y_size, x_size], the scale
                parameter of the Gaussian kernel.
            sigma: Tensor of shape [B, y_size], the magnitude
                of the std.
            sigma_noise: Float, std of the noise that we add for stability.

            Returns:
            The kernel, a float tensor of shape
            [B, y_size, num_total_points, num_total_points].
    
        """
        num_total_points = X.size()[1]

        #X.size(): [B, num_total_points, x_size]
        x1 = X.unsqueeze(1)# [B, 1, num_total_points, x_size]
        x2 = X.unsqueeze(2)# [B, num_total_points, 1, x_size]
        diff = x1 - x2 # [B, num_total_points, num_total_points, x_size]

        # [B, y_size, num_total_points, num_total_points, x_size]
        # None indexing [None, :] acts like tensor.unsqueeze(dim)
        norm = torch.square(diff[:, None, :, :, :] / l1[:, :, None, None, :])
        
        # [B, data_size, num_total_points, num_total_points]
        norm = norm.sum(-1) # one data point per row 

        # [B, y_size, num_total_points, num_total_points]
        kernel = torch.square(sigma)[:, :, None, None] * torch.exp(-0.5 * norm)
        
        # Add some noise to the diagonal to make the cholesky work.
        kernel.add_(torch.zeros_like(kernel).fill_diagonal_(noise**2))

        return kernel

    def generate_curves(self):
        """Builds the op delivering the data.

        Generated functions are `float32` with x values between -2 and 2.
        
        Returns:

        """

        num_context = 

        if self.testing
            num_target = 400
            num_total_points = num_target
            X = torch.range(-2, 2, 1. / 100).unsqueeze(0).expand(self.batch_size, -1) 
            # attention! returns view - copy necessary if in place operations are used 
            
            X = 
        else 
            num_target = 
            num_total_points = num_context + num_target
            X = torch.Tensor([self._batch_size, num_total_points, self._x_size]).uniform_(-2, 2)



        #set Kernel parameters randomly for every batch
        if self.random_params:
            length = torch.Tensor(self.batch_size, self.y_size, self.x_size).uniform_(0.1, self.length_scale)
            sigma = torch.Tensor(self.batch_size, self.y_size, self.x_size).uniform_(0.1, self.sigma_scale)
        
        else
        #use the same Kernel parameters for every batch
            length = torch.ones(self.batch_size, self.y_size, self.x_size).mul_(self.length_scale)
            sigma = torch.ones_like(length).mul_(self.sigma_scale)

        kernel = self._kernel(X, length, sigma) #TODO check dtype
        cholesky = kernel.cholesky() # TODO (maybe): change precision to float64 and cast to float32 afterwards
        y = cholesky.matmul(torch.randn(self.batch_size, self.y_size, num_total_points, 1)) #no mean assumption: y = mu + sigma*z~N(0,I) ~ c.L * rand_normal([0, 1]) with appropriate shape
        #TODO if runtime error: change dimension -1 of torch.randn_like(cholesky) to ?

        y = torch.transpose(y.squeeze(3)).permute(0, 2, 1) # possible error

        if self.testing:
            # Select the targets
            target_x = x
            target_y = y

            # Select the observations
            idx = torch.randperm(num_target)
            context_x = torch.index_select(x, 1, idx[:num_context] # tf.gather(x_values, idx[:num_context], axis=1)
            context_y = torch.index_select(y, 1, idx[:num_context]) # tf.gather(y_values, idx[:num_context], axis=1)
        
        else
            # Select the targets which will consist of the context points as well as
            # some new target points
            target_x = x[:, :num_target + num_context, :]
            target_y = y[:, :num_target + num_context, :]

            # Select the observations
            context_x = x[:, :num_context, :]
            context_y = y[:, :num_context, :]

    query = ((context_x, context_y), target_x)

    return query



# %%
