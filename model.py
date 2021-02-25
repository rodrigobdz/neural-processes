class NeuralProcess(nn.Module):

    def __init__(self, in_features, encoder_out, decoder_out, h_size, mc_size):
        super(NeuralProcess, self).__init__()
        # self._in_features = in_features
        # self._encoder_out = encoder_out
        # self._decoder_out = decoder_out
        # self._h_size = h_size

        self._mc_size = mc_size
        self._encoder = Encoder(in_features, encoder_out, h_size)
        self._decoder = Decoder(in_features, decoder_out, h_size)

    def forward(self, context_x, context_y, target_x, target_y=None):

        # q_prior will alyways be defined
        q_prior = self._encoder(context_x, context_y)

        #train time behaviour
        if target_y is not None:
            q_posterior = self._encoder(target_x, target_y)
            z = q_posterior.rsample([mc_size]) #rsample() takes care of rep. trick (z = µ + σ * I * ϵ , ϵ ~ N(0,1))

            # monte carlo sampling for integral over logp
            # z will be concatenate to every x_i and therefore must match
            # dimensionality of x_i
            z = z[:, :, None, :].expand(-1, -1, target_x.shape[1], -1)
            z = z.permute(1, 0, 2, 3)
            target_x = target_x[:, None, :, :].expand(-1, self._mc_size, -1, -1)


        #test time behaviour
        else:
            z = q_prior.rsample() #rsample() takes care of rep. trick (z = µ + σ * I * ϵ , ϵ ~ N(0,1))
            z = z[:, None, :].expand(-1, target_x.shape[1], -1)



        mu, sigma, distr = self._decoder(target_x, z)

        train = target_y is not None # true at train time
        q = (q_prior, q_posterior) if train else q_prior


        return (mu, sigma, distr), q