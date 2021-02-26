import matplotlib.pyplot as plt
import torch
from torch import distributions
from torch import nn
from torch import optim


# cast tensors to GPU if available
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# NP default parameter settings
batch_size = 16
max_num_context = 50
x_size = 1
y_size = 1
l_length = 0.6
sigma_length = 1.

num_iter = 100001
save_iter = 10000 # results printed every save_iter-th iteration


# generate training data
gp = GPCurves(batch_size, max_num_context, x_size, y_size, l_length, sigma_length, dev='cuda')
train_list = [gp.generate_curves() for _ in range(num_iter)]


# generate test data
batch_size = 1
gptest = GPCurves(batch_size, max_num_context, testing=True, dev='cuda')
test_list = [gptest.generate_curves()] # only one test curve is generated
# Use num_test_curves if more test curves are desired
# If so, adaption of fit function necessary
# num_test_curves = 100
# test_list = [gptest.generate_curves(num_test_curves)] # only one test curve is generated


# define model
in_features = 1
h_size = 128
encoder_out = [128, 256, 512, 1024] # [h_size]*4
decoder_out = [512, 256] + [2] # [128]*2 + [2]
mc_size = 1

np = NeuralProcess(in_features, encoder_out, decoder_out, h_size, mc_size)
np.to(dev)

# set optimi
opt = optim.Adam(np.parameters(), lr=5e-5)


# train np
mu, sigma = fit(niter, save_iter, np, opt, train_list, test_list)