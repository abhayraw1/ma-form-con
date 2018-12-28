import numpy as np
from util import info

class Noise(object):

    def __init__(self, delta, sigma, ou_a, ou_mu):
        # Noise parameters
        self.delta = delta
        self.sigma = sigma
        self.ou_a = ou_a
        self.ou_mu = ou_mu
        self.ou_lvl = np.zeros(self.ou_mu.shape)

    def brownian_motion_log_returns(self):
        sqrt_delta_sigma = np.sqrt(self.delta) * self.sigma
        return np.random.normal(loc=0, scale=sqrt_delta_sigma, size=None)

    def __call__(self):
        drift = self.ou_a * (self.ou_mu - self.ou_lvl) * self.delta
        randomness = self.brownian_motion_log_returns()
        self.ou_lvl += drift + randomness
        # info.out("{} {}".format(self.ou_lvl, id(self)))
        return self.ou_lvl
