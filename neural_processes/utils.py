#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as _plt
import torch as _torch
from torch import distributions as _distributions

def loss(distr, target_y, prior, posterior, mc_size):

    target_y = target_y[:, None, :, :].expand(-1, mc_size, -1, -1)
    logp = distr.log_prob(target_y).sum(dim=2, keepdims=True).mean(dim=1).squeeze()

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

        loss = loss(predict_distr, target_y, prior, posterior, np._mc_size)
        loss.backward()
        opt.step()
        opt.zero_grad()

        if i % save_iter == 0:
          np.eval()
          with torch.no_grad():
            # (mu, sigma, _), _ = np(context_x, context_y, target_x)
            # plot_functions(target_x, target_y, context_x, context_y, mu, sigma)

            # No target_y available at test time
            context_x, context_y, target_x, target_y = query_test[0]
            (mu, sigma, predict_distr), q = np(context_x, context_y, target_x)

            print(f'Iteration: {i}, loss: {loss}')
            plot_1d(context_x.cpu(), context_y.cpu(), target_x.cpu(), target_y.cpu(), mu.cpu(), sigma.cpu())

    return mu, sigma



# taken from colab NP
def plot_1d(context_x, context_y, target_x, target_y, pred_y, std):
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



def plot_2d(context_x, context_y, target_x, target_y, pred_y, std):
    return None


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


def unwrap(data, list_, batch_size):
    """unwrappes data which is present in format [batch_size, num_points, data_size]"""

    for i, d in enumerate(data):
          for j, x in enumerate(d):
              list_[j].append(x.to(dev)) # torch.reshape(x, (z * y, 1)))

def preprocess(gp, niter, batch_size):

    list_ = [[] for _ in range(niter * batch_size)]

    start = 0
    stop = 1
    for i in range(niter):
        # data.shape = [batch_size, no_of_elements_representing_one_function/curve, dim_of_x=1]
        data = gp.generate_curves()
        unwrap(data, list_[start*batch_size : stop*batch_size], batch_size)

        start += 1
        stop += 1
    return list_
