#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch import nn

from .plot import plot_results, gen_img

# Xavier_Uniform weight init
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def preprocess_mnist(Y, train=True):

    max_num_context = 400
    batch_size = Y.shape[0]

    # [batch_size, 1, 28, 28] -> [batch_size, 28, 28] (one channel img)
    Y = Y.squeeze(1) - .5 # mapping y to range [-.5, .5] (ANP paper); mapping to [0, 1] was done while loading mnist

    num_context = torch.randint(3, max_num_context+1, [])

    if train:
        num_target = torch.randint(0, max_num_context - num_context+1, [])
    else:
        num_context = torch.randint(100, max_num_context+1, []) # min 100 context points (while adjusting model)
        num_target = 28*28 - num_context # index and number of points


    num_total_points = num_context + num_target

    target_idx = torch.stack([torch.randperm(28*28)[:num_total_points] for _ in range(batch_size)]) # target includes context points; sampling without replacement
    target_row, target_col = unravel(target_idx) # map linear indices to Cartesian coordinates
    target_idx = torch.stack((target_row, target_col), dim=1) # stack indices together s.t. idx = [ [[row_ind_A], [col_ind_A]], ... [[row_ind_Z], [col_ind_Z]] ]

    target_x = target_idx.permute(0, 2, 1) # [B, num_total_point, dim]
    target_x = (target_x / 27) * 2 - 1 # rescaling x to range [-1, 1] (ANP paper)


    # [B, num_total_point, 1]
    target_y = torch.cat([y[i.tolist()] for y, i in zip(Y, target_idx)]).reshape(batch_size, num_total_points, 1)

    context_x = target_x[:, :num_context, :] # [B, num_context, dim]
    context_y = target_y[:, :num_context, :] # Y[context_idx.tolist()] # [B, num_context, 1]

    # dev is a global variable
    dev = 'cpu' if dev is None else dev

    context_x = context_x.to(dev)
    context_y = context_y.to(dev)
    target_x = target_x.to(dev)
    target_y = target_y.to(dev)

    return context_x, context_y, target_x, target_y



def unravel(idx):
    """
    pytorch will apply function vectorization, s.t. batch processing is possible
    unravel linear index [0, 28*28) to cartesian coordinates
    """

    col = idx % 28
    row = idx // 28

    return row, col


def rescale(x, y):
    """
    rescale coordinates to range [0, 28*28)
    rescale y to [0, 1]
    """

    scale_x = 27

    new_y = y + .5
    new_x = (x + 1).div(2) * scale_x

    new_x = new_x.round() # new_x.long() results in wrong positions therefore round
    new_x = new_x.long()

    return new_x, new_y



# set batch_size of test_generator to 1 if not already
def reconstruct(np_img, num_img, num_samples, test_generator, num_context = [10, 100, 300, 784], random_order = True):

    plots = [[] for _ in range(num_img)]

    for i, val in enumerate(test_generator):
        if i == num_img:
            break

        img, _ = val

        for context in num_context:
            context_x, context_y, target_x, _ = sample_context_target(img, context, random_order)

            # sample different zs to show variability given same context
            q = np_img._np._encoder(context_x, context_y)
            for _ in range(num_samples):
                z = q.sample()
                z = z[:, None, :].expand(-1, target_x.shape[1], -1)

                mu, _, _ = np_img._np._decoder(target_x, z)

                plot = gen_img(context_x[0], context_y[0], target_x[0], mu[0])
                plots[i].append(plot)

        with torch.no_grad():
            plot_results(plots, num_img, num_samples, num_context)


def sample_context_target(img, num_context, random_order):

    batch_size = img.shape[0]
    img = img.squeeze(1)

    # [batch_size, 1, 28, 28] -> [batch_size, 28, 28] (one channel img)
    Y = img - .5 # mapping y to range [-.5, .5] (ANP paper); mapping to [0, 1] was done while loading mnist

    num_target = 28*28 - num_context # index and number of points
    num_total_points = num_context + num_target

    if random_order:
        # sample random pixels as context points
        target_idx = torch.stack([torch.randperm(28*28)[:num_total_points] for _ in range(batch_size)]) # target includes context points; sampling without replacement

    else:
        # sample pixels in ordered fashion from Cartesian coordinates [0, 0], [0, 1], ..., [27, 27] as context points
        target_idx = torch.stack([torch.arange(28*28) for _ in range(batch_size)])

    target_row, target_col = unravel(target_idx) # map linear indices to Cartesian coordinates
    target_idx = torch.stack((target_row, target_col), dim=1) # stack indices together s.t. idx = [ [[row_ind_A], [col_ind_A]], ... [[row_ind_Z], [col_ind_Z]] ]

    target_x = target_idx.permute(0, 2, 1) # [B, num_total_point, dim]
    target_x = (target_x / 27) * 2 - 1 # rescaling x to range [-1, 1] (ANP paper)


    # [B, num_total_point, 1]
    target_y = torch.cat([y[i.tolist()] for y, i in zip(Y, target_idx)]).reshape(batch_size, num_total_points, 1) # TODO

    context_x = target_x[:, :num_context, :] # [B, num_context, dim]
    context_y = target_y[:, :num_context, :] # Y[context_idx.tolist()] # [B, num_context, 1]

    return context_x, context_y, target_x, target_y

