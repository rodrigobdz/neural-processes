#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import matplotlib.pyplot as _plt


def map_to_img(xc, yc, xt, yt, dev):
    """
    input is of shape as NP input excluding batch_dim, s.t. [num_points, [row_idx, col_idx]]
    returns image of shape [1, 28, 28]
    """

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


def rescale(x, y, dev):
    """
    rescale coordinates to range [0, 28*28)
    rescale y to [0, 1]
    """
    scale_x = 28 * 28 - 1

    new_y = y + .5
    new_y = new_y.to(dev)

    new_x = (x + 1).div(2) * scale_x
    new_x = new_x.round()
    new_x = new_x.long().to(dev)

    return new_x, new_y


def plot_1d(context_x, context_y, target_x, target_y, pred_y, std):
    # Taken from deepmind's colab
    """Plots the predicted mean and variance and the context points.

    Args:
      target_x: An array of shape [B,num_targets,1] that contains the
          x values of the target points.
      target_y: An array of shape [B,num_targets,1] that contains the
          y values of the target points.
      context_x: An array of shape [B,num_contexts,1] that contains
          the x values of the context points.
      context_y: An array of shape [B,num_contexts,1] that contains
          the y values of the context points.
      pred_y: An array of shape [B,num_targets,1] that contains the
          predicted means of the y values at the target points in target_x.
      std: An array of shape [B,num_targets,1] that contains the
          predicted std dev of the y values at the target points in target_x.
    """
    # Plot everything
    _plt.plot(target_x[0], pred_y[0], 'b', linewidth=2)
    _plt.plot(target_x[0], target_y[0], 'k:', linewidth=2)
    _plt.plot(context_x[0], context_y[0], 'ko', markersize=10)
    _plt.fill_between(
        target_x[0, :, 0],
        pred_y[0, :, 0] - std[0, :, 0],
        pred_y[0, :, 0] + std[0, :, 0],
        alpha=0.2,
        facecolor='#65c9f7',
        interpolate=True)

    # Make the plot pretty
    _plt.yticks([-2, 0, 2], fontsize=16)
    _plt.xticks([-2, 0, 2], fontsize=16)
    _plt.ylim([-2, 2])
    _plt.grid('off')
    ax = _plt.gca()
    _plt.show()


def plot_2d(context_x, context_y, target_x, target_y, data_generator=None, dev='cpu'):
    # undo scaling
    xc, yc = rescale(context_x, context_y, dev)
    xt, yt = rescale(target_x, target_y, dev)
    print(xc.shape[0])
    print(xt.shape[0])

    # map_to_img of shape 28*28
    img = map_to_img(xc, yc, xt, yt, dev)

    # plot prediction/completion
    plt.imshow(img.to('cpu'), cmap='gray')
    plt.title('prediction')
    plt.show()

    # plot original
    if data_generator is not None:
        for Y, label in data_generator:
            plt.imshow(Y[0].squeeze().to('cpu'), cmap='gray')
            plt.title(f'Original: {label[0]}')
            plt.show()
            break  # print first image


def plot_functions(target_x, target_y, context_x, context_y, pred_y, std):
    """Plots the predicted mean and variance and the context points.

    Args:
      target_x: An array of shape [B,num_targets,1] that contains the
          x values of the target points.
      target_y: An array of shape [B,num_targets,1] that contains the
          y values of the target points.
      context_x: An array of shape [B,num_contexts,1] that contains
          the x values of the context points.
      context_y: An array of shape [B,num_contexts,1] that contains
          the y values of the context points.
      pred_y: An array of shape [B,num_targets,1] that contains the
          predicted means of the y values at the target points in target_x.
      std: An array of shape [B,num_targets,1] that contains the
          predicted std dev of the y values at the target points in target_x.
    """
    # Plot everything
    _plt.plot(target_x, pred_y[0].unsqueeze(-1), 'b', linewidth=2)
    _plt.plot(target_x, target_y, 'k:', linewidth=2)
    _plt.plot(context_x, context_y, 'ko', markersize=10)
    _plt.fill_between(
        target_x,
        pred_y[0].unsqueeze(-1) - std[0],
        pred_y[0].unsqueeze(-1) + std[0],
        alpha=0.2,
        facecolor='#65c9f7',
        interpolate=True)

    # Make the plot pretty
    _plt.yticks([-2, 0, 2], fontsize=16)
    _plt.xticks([-2, 0, 2], fontsize=16)
    _plt.ylim([-2, 2])
    _plt.grid('off')
    ax = _plt.gca()
    _plt.show()


def plot_functions2(target_x, target_y, context_x, context_y, pred_y, std):
    """Plots the predicted mean and variance and the context points.

    Args:
      target_x: An array of shape [B,num_targets,1] that contains the
          x values of the target points.
      target_y: An array of shape [B,num_targets,1] that contains the
          y values of the target points.
      context_x: An array of shape [B,num_contexts,1] that contains
          the x values of the context points.
      context_y: An array of shape [B,num_contexts,1] that contains
          the y values of the context points.
      pred_y: An array of shape [B,num_targets,1] that contains the
          predicted means of the y values at the target points in target_x.
      std: An array of shape [B,num_targets,1] that contains the
          predicted std dev of the y values at the target points in target_x.
    """
    # Plot everything
    _plt.plot(target_x[0], pred_y[0], 'b', linewidth=2)
    _plt.plot(target_x[0], target_y[0], 'k:', linewidth=2)
    _plt.plot(context_x[0], context_y[0], 'ko', markersize=10)
    _plt.fill_between(
        target_x[0, :, 0],
        pred_y[0, :, 0] - std[0, :, 0],
        pred_y[0, :, 0] + std[0, :, 0],
        alpha=0.2,
        facecolor='#65c9f7',
        interpolate=True)

    # Make the plot pretty
    _plt.yticks([-2, 0, 2], fontsize=16)
    _plt.xticks([-2, 0, 2], fontsize=16)
    _plt.ylim([-2, 2])
    _plt.grid('off')
    ax = _plt.gca()
    _plt.show()
