#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as _torch
from torch import distributions as _distributions
from .plot import plot_1d, plot_2d, plot_functions, plot_functions2


def loss(distr, target_y, prior, posterior, mc_size):

    target_y = target_y[:, None, :, :].expand(-1, mc_size, -1, -1)
    logp = distr.log_prob(target_y).sum(
        dim=2, keepdims=True).mean(dim=1).squeeze()

    # analytic solution exists since two MvGaussians are used
    kl = _distributions.kl_divergence(posterior, prior)

    # optimiser uses gradient descent but
    # ELBO should be maximized: therefore -loss
    loss = -_torch.mean(logp - kl)

    return loss


def fit(niter, save_iter, np, opt, train_set, query_test):

    for i in range(niter):
        np.train()
        context_x, context_y, target_x, target_y = train_set[i]
        distr_tuple, q = np(context_x, context_y, target_x, target_y)

        predict_distr = distr_tuple[2]
        prior = q[0]
        posterior = q[1]

        training_loss = loss(predict_distr, target_y,
                             prior, posterior, np._mc_size)
        training_loss.backward()
        opt.step()
        opt.zero_grad()

        if i % save_iter == 0:
            np.eval()
            with _torch.no_grad():
                # (mu, sigma, _), _ = np(context_x, context_y, target_x)
                # plot_functions(target_x, target_y, context_x, context_y, mu, sigma)

                # No target_y available at test time
                context_x, context_y, target_x, target_y = query_test[0]
                (mu, sigma, predict_distr), q = np(
                    context_x, context_y, target_x)

                print(f'Iteration: {i}, loss: {training_loss}')
                plot_1d(context_x.cpu(), context_y.cpu(), target_x.cpu(),
                        target_y.cpu(), mu.cpu(), sigma.cpu())

    return mu, sigma


# rescale coordinates to range [0, 28*28)
# rescale y to [0, 1]

def rescale(x, y, dev):
    scale_x = 28 * 28 - 1

    new_y = y + .5
    new_y = new_y.to(dev)

    new_x = (x + 1).div(2) * scale_x
    new_x = new_x.round()
    new_x = new_x.long().to(dev)

    return new_x, new_y

# input is of shape as NP input excluding batch_dim, s.t. [num_points, [row_idx, col_idx]]
# returns image of shape [1, 28, 28]


def map_to_img(xc, yc, xt, yt, dev):

    img = torch.zeros(28, 28)
    img = img.to(dev)

    num_context = xc.shape[0]

    # TODO use numpy/pytorch indexing style
    for i, idx in enumerate(xt):

        if i < num_context:
            y = yc[i]

        # for test reasons ignore predictions for known context_y and use true values
        else:
            y = yt[i]
        # y = yt[i]
        img[idx[0], idx[1]] = y

    return img


def unravel(idx):
    """
    pytorch will apply function vectorization, s.t. batch processing is possible
    unravel linear index [0, 28*28) to cartesian coordinates
    """

    col = idx % 28
    row = idx // 28

    return row, col
