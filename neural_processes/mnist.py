#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch as _torch
from .utils import unravel, loss
from .plot import plot_2d, rescale


def preprocess_mnist(data_generator, dev, train=True):
    """Prepare mnist train data beforehand"""

    list_ = []
    max_num_context = 400
    # highest possible linear index used for rescaling to range [-1, 1]
    rescale_x = 28*28-1

    for Y, _ in data_generator:

        batch_size = Y.shape[0]

        # [batch_size, 1, 28, 28] -> [batch_size, 28, 28] (one channel img)
        # mapping y to range [-.5, .5] (ANP paper); mapping to [0, 1] was done while loading mnist
        Y = Y.squeeze(1) - .5

        num_context = _torch.randint(3, max_num_context, [])

        if train:
            num_target = _torch.randint(0, max_num_context - num_context, [])
        else:
            # min 100 context points (while adjusting model)
            num_context = _torch.randint(100, max_num_context, [])
            num_target = 28*28 - num_context - 1  # index and number of points

        num_total_points = num_context + num_target

        # target includes context points; sampling without replacement
        target_idx = _torch.stack(
            [_torch.randperm(28*28-1)[:num_total_points] for _ in range(batch_size)])
        # map linear indices to Cartesian coordinates
        target_col, target_row = unravel(target_idx)
        # stack indices together s.t. idx = [ [[row_ind_A], [col_ind_A]], ... [[row_ind_Z], [col_ind_Z]] ]
        target_idx = _torch.stack((target_row, target_col), dim=1)

        target_x = target_idx.permute(0, 2, 1)  # [B, num_total_point, dim]
        # rescaling x to range [-1, 1] (ANP paper)
        target_x = (target_x / rescale_x) * 2 - 1

        # [B, num_total_point, 1]
        target_y = _torch.cat([y[i.tolist()] for y, i in zip(Y, target_idx)]).reshape(
            batch_size, num_total_points, 1)  # TODO find vectorized approach

        context_x = target_x[:, :num_context, :]  # [B, num_context, dim]
        # Y[context_idx.tolist()] # [B, num_context, 1]
        context_y = target_y[:, :num_context, :]

        context_x = context_x.to(dev)
        context_y = context_y.to(dev)
        target_x = target_x.to(dev)
        target_y = target_y.to(dev)

        list_.append([context_x, context_y, target_x, target_y])

    return list_


def preprocess_mnist2(Y, dev, train=True):

    list_ = []
    max_num_context = 400
    # highest possible linear index used for rescaling to range [-1, 1]
    rescale_x = 28*28-1

    batch_size = Y.shape[0]

    # [batch_size, 1, 28, 28] -> [batch_size, 28, 28] (one channel img)
    # mapping y to range [-.5, .5] (ANP paper); mapping to [0, 1] was done while loading mnist
    Y = Y.squeeze(1) - .5

    num_context = _torch.randint(3, max_num_context, [])

    if train:
        num_target = _torch.randint(0, max_num_context - num_context, [])
    else:
        # min 100 context points (while adjusting model)
        num_context = _torch.randint(100, max_num_context, [])
        num_target = 28*28 - num_context - 1  # index and number of points

    num_total_points = num_context + num_target

    # target includes context points; sampling without replacement
    target_idx = _torch.stack(
        [_torch.randperm(28*28-1)[:num_total_points] for _ in range(batch_size)])
    # map linear indices to Cartesian coordinates
    target_col, target_row = unravel(target_idx)
    # stack indices together s.t. idx = [ [[row_ind_A], [col_ind_A]], ... [[row_ind_Z], [col_ind_Z]] ]
    target_idx = _torch.stack((target_row, target_col), dim=1)

    target_x = target_idx.permute(0, 2, 1)  # [B, num_total_point, dim]
    # rescaling x to range [-1, 1] (ANP paper)
    target_x = (target_x / rescale_x) * 2 - 1

    # [B, num_total_point, 1]
    target_y = _torch.cat([y[i.tolist()] for y, i in zip(Y, target_idx)]).reshape(
        batch_size, num_total_points, 1)  # TODO find vectorized approach

    context_x = target_x[:, :num_context, :]  # [B, num_context, dim]
    # Y[context_idx.tolist()] # [B, num_context, 1]
    context_y = target_y[:, :num_context, :]

    context_x = context_x.to(dev)
    context_y = context_y.to(dev)
    target_x = target_x.to(dev)
    target_y = target_y.to(dev)

    list_.extend([context_x, context_y, target_x, target_y])

    return list_


def fit_mnist(epochs, niter, save_epoch, np, opt, train_generator, test_generator, dev):

    query_test = preprocess_mnist(test_generator, dev, train=False)
    # train_generator = torch.utils.data.DataLoader(train_mnist, **params)

    for j in range(epochs):
        train_set = preprocess_mnist(train_generator, dev)

        for i in range(niter):
            # np.train()
            context_x, context_y, target_x, target_y = train_set[i]
            distr_tuple, q = np(context_x, context_y, target_x, target_y)

            predict_distr = distr_tuple[2]
            prior = q[0]
            posterior = q[1]

            loss_ = loss(predict_distr, target_y,
                         prior, posterior, np._mc_size)
            loss_.backward()
            opt.step()
            opt.zero_grad()

            if i % 1000 == 0:
                # print(f'Iteration: {i}, loss: {loss_.detach()}')
                with torch.no_grad():

                    context_x, context_y, target_x, target_y = query_test[0]
                    (mu, sigma, predict_distr), q = np(
                        context_x, context_y, target_x)

                    print(f'Epoch: {j}, Iteration: {i}, loss: {loss_}')

        if j % save_epoch == 0:
            # np.eval()
            with torch.no_grad():
                # (mu, sigma, _), _ = np(context_x, context_y, target_x)
                # plot_functions(target_x, target_y, context_x, context_y, mu, sigma)

                # No target_y available at test time
                context_x, context_y, target_x, target_y = query_test[0]
                (mu, sigma, predict_distr), q = np(
                    context_x, context_y, target_x)

                print(f'Iteration: {i}, loss: {loss_}')
                plot_2d(context_x[0], context_y[0], target_x[0],
                        mu[0], data_generator=test_generator, dev=dev)

    return mu, sigma


def fit_mnist2(epochs, save_epoch, np, opt, train_generator, test_generator, dev):

    query_test = preprocess_mnist(test_generator, dev, train=False)
    # train_generator = _torch.utils.data.DataLoader(train_mnist, **params)

    for j in range(epochs):
        for i, (Y, _) in enumerate(train_generator):
            train_set = preprocess_mnist2(Y, dev, train=True)

            context_x, context_y, target_x, target_y = train_set
            distr_tuple, q = np(context_x, context_y, target_x, target_y)

            predict_distr = distr_tuple[2]
            prior = q[0]
            posterior = q[1]

            loss_ = loss(predict_distr, target_y,
                         prior, posterior, np._mc_size)
            loss_.backward()
            opt.step()
            opt.zero_grad()

            if i % 1000 == 0:
                # print(f'Iteration: {i}, loss: {loss_.detach()}')
                with _torch.no_grad():

                    context_x, context_y, target_x, target_y = query_test[0]
                    (mu, sigma, predict_distr), q = np(
                        context_x, context_y, target_x)

                    print(f'Epoch: {j}, Iteration: {i}, loss: {loss_}')

        if j % save_epoch == 0:
            # np.eval()
            with _torch.no_grad():
                # (mu, sigma, _), _ = np(context_x, context_y, target_x)
                # plot_functions(target_x, target_y, context_x, context_y, mu, sigma)

                # No target_y available at test time
                context_x, context_y, target_x, target_y = query_test[0]
                (mu, sigma, predict_distr), q = np(
                    context_x, context_y, target_x)

                print(f'Iteration: {i}, loss: {loss_}')
                plot_2d(context_x[0], context_y[0], target_x[0],
                        mu[0], data_generator=test_generator, dev=dev)

    return mu, sigma
