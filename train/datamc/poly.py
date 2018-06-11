#!/usr/bin/env python

from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from subtlenet.backend.keras_objects import *
from subtlenet.backend.losses import *
from subtlenet.backend.layers import *
from tensorflow.python.framework import graph_util, graph_io
import os
import numpy as np
from sys import stdout, exit
from tqdm import tqdm

from subtlenet import utils
utils.set_processor('cpu')

sqr = np.square
NZBINS = 20; ZLO = 0; ZHI = 3
ZBIN = False
lmbda = 150
D = 2

def _make_parent(path):
    os.system('mkdir -p %s'%('/'.join(path.split('/')[:-1])))


class PlotCfg(object):
    def __init__(self, name, binning, fn, classes=[0],
                 weight_fn=None, cut_fn=None, xlabel=None, ylabel=None):
        self._fn = fn
        self._weight_fn = weight_fn
        self._cut_fn = cut_fn
        self.name = name
        self._bins = binning
        self.hists = {c:utils.NH1(binning) for c in classes}
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.classes = classes
    def clear(self):
        for _,h in self.hists.iteritems():
            h.clear()
    def add_data(self, data):
        x = self._fn(data)
        weight = self._weight_fn(data) if self._weight_fn is not None else None
        cut = self._cut_fn(data) if self._cut_fn is not None else np.ones(x.shape[0]).astype(bool)
        for c in self.classes:
            if len(self.classes) == 1:
                mask = cut
            else:
                mask = np.logical_and(cut, data[1] == c)
            w = weight[mask] if weight is not None else None
            self.hists[c].fill_array(x[mask], weights=w)


class Data(object):
    def __init__(self, n_input=1):
        self.n_input = n_input
        self.plotter = utils.Plotter()
        self.roc = utils.Roccer()
    def gen(self, N):
#        x = np.random.uniform(size=N)
#        for _ in xrange(3):
#            x = np.maximum(x, np.random.uniform(size=N))
#        for _ in xrange(3):
#            x = np.minimum(x, np.random.uniform(size=N))
#        x = np.random.normal(loc=0.5,scale=0.5,size=N)
        x = np.random.exponential(size=N)
        mask = np.logical_and(x>0, x<1)
        return x[mask], np.random.uniform(size=np.sum(mask))
    def plot(self, outpath, cfgs, N, order=None, scale=True):
        outpath += '/'
        _make_parent(outpath)
        _batch_size = 10000
        for _ in xrange(N / _batch_size):
            for cfg in cfgs:
                cfg.add_data(self.gen(_batch_size))
        for cfg in cfgs:
            self.plotter.clear()
            if order is None:
                order = cfg.hists.keys()
            for i,label in enumerate(order):
                if label not in cfg.hists:
                    continue
                if scale:
                    cfg.hists[label].scale()
                self.plotter.add_hist(cfg.hists[label], str(label), i)
            self.plotter.plot(xlabel=cfg.xlabel,
                              ylabel=cfg.ylabel,
                              output=outpath+'/'+cfg.name)



if __name__ == '__main__':
    figsdir = '/home/snarayan/public_html/figs/adv/poly/v0/'
    from argparse import ArgumentParser
    parser = ArgumentParser()
    args = parser.parse_args()

    data = Data()
    X = Input(shape=(1,), name='X')
    Y = ConvexPolyLayer(D, alpha=0)(X)
    model = Model(inputs=[X], outputs=[Y])
    model.compile(loss=[pred_loss],
                  optimizer=Adam())
    model.summary()

    data_fn = data.gen
    batch = 1000
#    print data.gen(1)
#    print '#'*20
#    l =  model.test_on_batch(*data.gen(2))
#    print '#'*20
#    print l
#    exit(-1)
    for _ in tqdm(range(5000)):
        x, y = data.gen(batch)
        model.train_on_batch(x, y)

    print model.layers[-1].poly_coeffs
    coeffs = [x[0] for x in model.layers[-1].poly_coeffs]
    integral = model.layers[-1].integral
    poly_fn = np.polynomial.polynomial.Polynomial(coeffs)
#    poly_scaled = lambda x : poly_fn(x) / integral
#    minval = np.min(poly_scaled(np.linspace(0,1,1000)))
#    poly = lambda x : poly_scaled(x) - minval
    poly = lambda x : poly_fn(x) / integral

    x = np.linspace(0, 1, 5)
    print 'x'
    print x
    print 'predict'
    print model.predict(x)
    print 'poly'
    print poly(x)
    print 'coeffs'
    print coeffs
    print 'integral'
    print integral

    x = PlotCfg('x', np.linspace(0, 1, 40),
                lambda x : x[0])
    pred = PlotCfg('pred', np.linspace(0, 1, 40),
                   lambda x : x[1],
                   weight_fn =  lambda x : poly(x[1])  )
    data.plot(figsdir, [x, pred], 100000, scale=False)

#    x = {i : PlotCfg('x%i'%i, np.linspace(-1, 3, 40),
#                     lambda x, i=i : x[0][:,i])
#         for i in xrange(D)}
#    nn = PlotCfg('nn', np.linspace(0,1,40),
#                 lambda x : model.predict(1, x[0]))
#    nn_noA = PlotCfg('nn_noA', np.linspace(0,1,40),
#                     lambda x : model_noA.predict(1, x[0]))
#    z_binning = np.linspace(0, NZBINS, NZBINS) if ZBIN else np.linspace(0, 1, NZBINS)
#    z = PlotCfg('z', z_binning,
#                lambda x : x[2])
#
#    data.plot(figsdir, x.values() + [nn, nn_noA, z], 100000, order=range(2))
#
#    cut_nn = nn.hists[0].quantile(0.95)
#    zcut = PlotCfg('zcut', z_binning,
#                   lambda x : x[2],
#                   cut_fn=lambda x : (model.predict(1, x[0])>cut_nn))
#    cut_nnnoA = nn_noA.hists[0].quantile(0.95)
#    zcut_noA = PlotCfg('zcut_noA', z_binning,
#                       lambda x : x[2],
#                       cut_fn=lambda x : (model_noA.predict(1, x[0])>cut_nnnoA))
#
#    data.plot(figsdir, [zcut, zcut_noA], 100000, order=range(2))
