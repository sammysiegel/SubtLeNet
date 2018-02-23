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
    bias = np.clip(_normal(mu, sigma) + 1, 0, 2) # treat it as a fractional resolution
    arr[:,:,:4] *= bias.reshape(arr.shape[:-1] + (-1,)) 
    return arr 
    

class _Callable(object):
    def __init__(self, f):
        if isinstance(f, Number):
            self.f = lambda x : f
        else:
            self.f = f
    def __call__(self, *args):
        return self.f(*args)

class CaloSmear(object):
    def __init__(self, c_mu, c_sigma, n_mu, n_sigma):
        self.c_mu = _Callable(c_mu)
        self.c_sigma = _Callable(c_sigma)
        self.n_mu = _Callable(n_mu)
        self.n_sigma = _Callable(n_sigma)
        self._n_mask = None

    @staticmethod
    def _is_neutral(ptype):
        # forgive me, for I have sinned 
        ptype = (2.5958684827557796 + (ptype * 2.2772611292950851)).astype(np.int)
        return np.logical_or(ptype == 0,                 # calo object
                             np.logical_or(ptype == 3,   # photon
                                           ptype == 4))  # neutral hadron
                
    def _mu(self, p4, ptype):
        return (self.n_mu(p4) * self._n_mask) + (self.c_mu(p4) * ~(self._n_mask))

    def _sigma(self, p4, ptype):
        return (self.n_sigma(p4) * self._n_mask) + (self.c_sigma(p4) * ~(self._n_mask))

    def __call__(self, arr):
        self._n_mask = CaloSmear._is_neutral(arr[:,:,6])
        return gauss(arr, self._mu, self._sigma)
        

