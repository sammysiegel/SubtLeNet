# smearing functions used for resolution and
# systematic uncertainty studies

import numpy as np 
from numbers import Number

_normal = np.random.normal

def gauss(arr, mu, sigma):
    # shape of arr is (n_batch, n_particles, n_features)
    if sigma == 0:
        arr[:,:,:4] += mu
        return arr 
    if isinstance(mu, Number):
        mu = mu * np.ones(arr.shape[:-1])
    else:
        mu = mu(arr[:,:,3], arr[:,:,6]) # f'n of particle energy and type 
    if isinstance(sigma, Number):
        sigma = sigma * np.ones(arr.shape[:-1])
    else:
        sigma = sigma(arr[:,:,3], arr[:,:,6]) # f'n of particle energy and type 
    bias = _normal(mu, sigma) + 1 # treat it as a fractional resolution
    arr[:,:,:4] *= bias.reshape(arr.shape[:-1] + (-1,)) 
    return arr 
    

# TODO - implement parameters as functions of p4
class CaloSmear(object):
    def __init__(self, c_mu, c_sigma, n_mu, n_sigma):
        self.c_mu = c_mu
        self.c_sigma = c_sigma
        self.n_mu = n_mu
        self.n_sigma = n_sigma

    @staticmethod
    def _is_neutral(ptype):
        return np.logical_or(ptype == 0,                 # calo object
                             np.logical_or(ptype == 3,   # photon
                                           ptype == 4))  # neutral hadron
                
    def _mu(self, p4, ptype):
        n = CaloSmear._is_neutral(ptype)
        return (self.n_mu * n) + (self.c_mu * ~n)

    def _sigma(self, p4, ptype):
        n = CaloSmear._is_neutral(ptype)
        return (self.n_sigma * n) + (self.c_sigma * ~n)

    def __call__(self, arr):
        return gauss(arr, self._mu, self._sigma)
        

