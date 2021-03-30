#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import matplotlib.pyplot as plt

from .utils import rescale


def map_to_img(xc, yc, xt, yt):
    """
    input is of shape as NP input excluding batch_dim, s.t. [num_points, [row_idx, col_idx]]
    returns image of shape [1, 28, 28]
    """

    img = torch.ones(28, 28)
    num_context = xc.shape[0]

    # TODO use numpy/pytorch indexing style
    for i, idx in enumerate(xt):

        if i < num_context:
            y = yc[i]
        else:
            y = yt[i]

        img[idx[0], idx[1]] = y

    return img



def gen_img(context_x, context_y, target_x, target_y):

    # undo scaling
    xc, yc = rescale(context_x, context_y)
    xt, yt = rescale(target_x, target_y)

    # map_to_img of shape 28*28
    img = map_to_img(xc, yc, xt, yt)

    return img


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
    plt.plot(target_x[0], pred_y[0], 'b', linewidth=2)
    plt.plot(target_x[0], target_y[0], 'k:', linewidth=2)
    plt.plot(context_x[0], context_y[0], 'ko', markersize=10)
    plt.fill_between(
        target_x[0, :, 0],
        pred_y[0, :, 0] - std[0, :, 0],
        pred_y[0, :, 0] + std[0, :, 0],
        alpha=0.2,
        facecolor='#65c9f7',
        interpolate=True)

    # Make the plot pretty
    plt.yticks([-2, 0, 2], fontsize=16)
    plt.xticks([-2, 0, 2], fontsize=16)
    plt.ylim([-2, 2])
    plt.grid('off')

    # ax = plt.gca() # currently don't know what this does
    plt.show()


def plot_2d(context_x, context_y, target_x, prediction, target=None, label=None):

    img = gen_img(context_x, context_y, target_x, prediction)

    # plot prediction/completion
    plt.imshow(img, cmap='gray')
    plt.title('prediction')
    plt.show()

    if target is not None and label is not None:
        plt.imshow(target.squeeze(), cmap='gray')
        plt.title(f'Original: {label}')
        plt.show()


def plot_results(plots, num_img, num_samples, num_context):

    fig, axs = plt.subplots(num_samples, num_img, figsize=(10, 10))

    for i in range(num_img):
        for j in range(num_samples):

            axs[i, j].imshow(plots[i][j], cmap='gray')
            # axs[i, j]. # TODO add num_context[i] to y-axis

    # plt.
    plt.show()


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
    plt.plot(target_x[0], pred_y[0], 'b', linewidth=2)
    plt.plot(target_x[0], target_y[0], 'k:', linewidth=2)
    plt.plot(context_x[0], context_y[0], 'ko', markersize=10)
    plt.fill_between(
        target_x[0, :, 0],
        pred_y[0, :, 0] - std[0, :, 0],
        pred_y[0, :, 0] + std[0, :, 0],
        alpha=0.2,
        facecolor='#65c9f7',
        interpolate=True)

    # Make the plot pretty
    plt.yticks([-2, 0, 2], fontsize=16)
    plt.xticks([-2, 0, 2], fontsize=16)
    plt.ylim([-2, 2])
    plt.grid('off')

    # ax = plt.gca()

    plt.show()
