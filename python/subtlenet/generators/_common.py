import numpy as np
from .. import config, utils
from ..backend import obj, smear
from os import environ
environ['KERAS_BACKEND'] = 'tensorflow'
from keras.utils import np_utils


class _Reshaper(object):
    def __init__(self, histpath):
        h = utils.NH1.load(histpath)
        self._nbins = h._content.shape[0]
        self._bw = h.bins[1] - h.bins[0] # assume constant
        self._transform = np.diag(h._content)
    def __call__(self, x):
        x = (x / self._bw).astype(int)
        x = np.clip(x, 0, self._nbins - 1)
        xoh = np_utils.to_categorical(x, self._nbins)
        scaled = np.dot(xoh, self._transform)
        return np.sum(scaled, axis=-1)


class _Generator(object):
    def __init__(self, 
                 collections,
                 partition, batch, 
                 components, 
                 repartition=True, 
                 window=False, reshape=False,
                 decorr_mass=False, decorr_pt=False,
                 normalize=False, **kwargs):
        self.partition = partition
        self.small_batch = max(1, int(batch / len(collections)))
        self.generators = {c:c.generator(components=components,
                                         partition=partition,
                                         batch=self.small_batch,
                                         repartition=repartition,
                                         normalize=normalize)
                            for c in collections}
        self.normalize = normalize
        self.decorr_mass = decorr_mass
        self.decorr_pt = decorr_pt
        self.window = window
        self.reshape = reshape
        self.collections = collections
    def _set_masks(self):
        self.variation_mask = {c:(c.label >= 0) for c in self.collections}
        self.train_mask = {c:(c.label <= 0) for c in self.collections}
    def _set_xforms(self):
        self.msd_index = config.gen_singletons['msd']
        self.pt_index = config.gen_singletons['pt']
        self.msd_norm_factor = 1. / config.max_mass 
        self.pt_norm_factor = 1. / (config.max_pt - config.min_pt)
        if config.n_decorr_bins > 1:
            def xform_mass(x):
                binned = (np.minimum(x, config.max_mass) 
                          * self.msd_norm_factor 
                          * (config.n_decorr_bins - 1)
                         ).astype(np.int)
                onehot = np_utils.to_categorical(binned, config.n_decorr_bins)
                return onehot
            def xform_pt(x):
                binned = (np.minimum(x-config.min_pt, config.max_pt-config.min_pt) 
                          * self.pt_norm_factor 
                          * (config.n_decorr_bins - 1)
                         ).astype(np.int)
                onehot = np_utils.to_categorical(binned, config.n_decorr_bins)
                return onehot
        else:
            def xform_mass(x):
                return np.minimum(x, config.max_mass) * self.msd_norm_factor         
            def xform_pt(x):
                return (np.minimum(x-config.min_pt, config.max_pt-config.min_pt) * self.pt_norm_factor)
        self.xform_mass = xform_mass
        self.xform_pt = xform_pt
    def _set_smearer(self, smear_params):
        if smear_params is not None:
            if len(smear_params) == 2:
                self.smearer = lambda x : smear.gauss(x, *smear_params)
            elif len(smear_params) == 4:
                self.smearer = smear.CaloSmear(*smear_params)
        else:
            self.smearer = None
    def _set_mass_reshaper(self):
        self.mass_reshaper = _Reshaper(environ['SUBTLENET'] + '/data/mass_scale.npy')

