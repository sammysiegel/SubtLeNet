#!/usr/bin/env python

from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from subtlenet.backend.keras_objects import *
from subtlenet.backend.losses import *
from tensorflow.python.framework import graph_util, graph_io
import os
import numpy as np
from sys import stdout
from tqdm import tqdm

from subtlenet import utils
utils.set_processor('cpu')

sqr = np.square
NZBINS = 20; ZLO = 0; ZHI = 3
ZBIN = False
lmbda = 150
D = 5

def _make_parent(path):
    os.system('mkdir -p %s'%('/'.join(path.split('/')[:-1])))


class PlotCfg(object):
    def __init__(self, name, binning, fn, classes=[0,1],
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
            mask = np.logical_and(cut, data[1] == c)
            w = weight[mask] if weight is not None else None
            self.hists[c].fill_array(x[mask], weights=w)


class Data(object):
    def __init__(self, n_input=D, n_output=1):
        self.n_input = n_input
        self.n_output = n_output
        self.plotter = utils.Plotter()
        self.roc = utils.Roccer()
    def gen(self, N, bin=True):
        # returns 2*N samples
        # first gaussians
        s_x = np.random.randn(N, self.n_input)*0.5 + 1
        b_x_g = np.random.randn(N, self.n_input-1)*0.5+0.25
        b_x_e = np.random.exponential(size=(N,1))
        b_x = np.concatenate([b_x_g, b_x_e], axis=-1)
        x = np.concatenate([s_x, b_x], axis=0)

        y = np.concatenate([np.ones(N), np.zeros(N)], axis=0)
        if bin:
            y = np_utils.to_categorical(y, 2)

        z = (x[:,-1] - ZLO) / (ZHI - ZLO)
        if ZBIN:
            z = (z * NZBINS).astype(int)
            z = np.clip(z, 0, NZBINS-1)
            if bin:
                z = np_utils.to_categorical(z, NZBINS)

        w = np.ones((2*N,))
        w_masked = np.concatenate([np.zeros((N,)), np.ones((N,))])

        idx = np.random.permutation(2*N)
        return x[idx], y[idx], z[idx], w[idx], w_masked[idx]
    def plot(self, outpath, cfgs, N, order=None, scale=True):
        outpath += '/'
        _make_parent(outpath)
        _batch_size = 10000
        for _ in xrange(N / _batch_size):
            for cfg in cfgs:
                cfg.add_data(self.gen(_batch_size, bin=False))
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


class DAModel(object):
    @staticmethod
    def _scale_loss(scale, loss):
        def _loss(y_true, y_pred):
            return scale * loss(y_true, y_pred)
        return _loss
    def __init__(self, n_inputs=D, n_targets=2, l=lmbda, losses=None, loss_weights=None):
        if losses is None:
            losses = [categorical_crossentropy]
            if ZBIN:
                losses.append(categorical_crossentropy)
            else:
                losses.append(huber)

        self.n_inputs = n_inputs
        self.n_targets = n_targets
        self.l = l

        X = Input(shape=(n_inputs,), name='X')
        h = Dense(2*n_inputs, activation='tanh', kernel_initializer='lecun_uniform')(X)
#        h = Dense(2*n_inputs, activation='relu', kernel_initializer='lecun_uniform')(h)
#        h = Dense(2*n_inputs, activation='relu', kernel_initializer='lecun_uniform')(h)
#        h = Dense(2*n_inputs, activation='relu', kernel_initializer='lecun_uniform')(h)
        Y = Dense(n_targets, activation='softmax', kernel_initializer='lecun_uniform')(h)

        h = Dense(2*n_inputs, activation='tanh', kernel_initializer='lecun_uniform')(Y)
#        h = Dense(2*n_inputs, activation='relu', kernel_initializer='lecun_uniform')(h)
#        h = Dense(2*n_inputs, activation='relu', kernel_initializer='lecun_uniform')(h)
#        h = Dense(2*n_inputs, activation='relu', kernel_initializer='lecun_uniform')(h)
        if ZBIN:
            Z = Dense(NZBINS, activation='softmax', kernel_initializer='lecun_uniform')(h)
        else:
            Z = Dense(1, activation='linear', kernel_initializer='lecun_uniform')(h)

        # discriminator
        self.D = Model(inputs=[X], outputs=[Y])
        self.D.compile(loss=[losses[0]],
                       optimizer=Adam())
        DAModel._sum('D', self.D)

        # adversary
        self.A = Model(inputs=[X], outputs=[Z])
        utils.freeze(self.A, True, self.D.layers) # un-freeze all in A except D
        self.A.compile(loss=[losses[1]],
                       optimizer=SGD())
        DAModel._sum('A', self.A)

        # D+A stack
        self.DAs = Model(inputs=[X], outputs=[Y,Z])
        utils.freeze(self.DAs, False, self.D.layers) # freeze all in DAs except D
        self.DAs.compile(loss=[losses[0],
                                DAModel._scale_loss(-l, losses[1])],
                         optimizer=Adam())
        DAModel._sum('DAs', self.DAs)

    @staticmethod
    def _sum(name, model):
        s = '====== Model %s ======'%name
        print s
        model.summary()
        print '=' * len(s)
    @staticmethod
    def _fit(model, x, y, w=None):
        model.fit(x, y, sample_weight=w, verbose=0)
    @staticmethod
    def _tob(model, x, y, w=None):
        model.train_on_batch(x, y, sample_weight=w)
    def train_D(self, data_fn, batch_size, **kwargs):
        data = data_fn(batch_size)
        DAModel._tob(self.D, data[0], data[1], **kwargs)
    def train_A(self, data_fn, batch_size, **kwargs):
        data = data_fn(batch_size)
        mask = data[-1] == 1
        DAModel._fit(self.A, data[0][mask], data[2][mask], **kwargs)
    def train_DAs(self, data_fn, batch_size, **kwargs):
        data = data_fn(batch_size)
        DAModel._tob(self.DAs, data[0], [data[1], data[2]], [data[3], data[4]], **kwargs)
    def save_as_keras(self, path):
        _make_parent(path)
        self.DAs.save(path)
        print 'Saved to',path
    def save_as_tf(self,path):
        _make_parent(path)
        sess = K.get_session()
        print [l.op.name for l in self.DAs.inputs],'->',[l.op.name for l in self.DAs.outputs]
        graph = graph_util.convert_variables_to_constants(sess,
                                                          sess.graph.as_graph_def(),
                                                          [n.op.name for n in self.DAs.outputs])
        p0 = '/'.join(path.split('/')[:-1])
        p1 = path.split('/')[-1]
        graph_io.write_graph(graph, p0, p1, as_text=False)
        print 'Saved to',path
    def predict(self, cls, *args, **kwargs):
        return self.D.predict(*args, **kwargs)[:,cls]


if __name__ == '__main__':
    figsdir = '/home/snarayan/public_html/figs/adv/v0/'
    from argparse import ArgumentParser
    parser = ArgumentParser()
    args = parser.parse_args()

    data = Data()
    model = DAModel()
    model_noA = DAModel()

    data_fn = data.gen
    batch = 128
    model.train_D(data_fn, batch*4)
    model.train_A(data_fn, batch*4)
    for _ in tqdm(range(5000)):
        model.train_DAs(data_fn, batch)
        model.train_A(data_fn, batch)
        model_noA.train_D(data_fn, batch)

    x = {i : PlotCfg('x%i'%i, np.linspace(-1, 3, 40),
                     lambda x, i=i : x[0][:,i])
         for i in xrange(D)}
    nn = PlotCfg('nn', np.linspace(0,1,40),
                 lambda x : model.predict(1, x[0]))
    nn_noA = PlotCfg('nn_noA', np.linspace(0,1,40),
                     lambda x : model_noA.predict(1, x[0]))
    z_binning = np.linspace(0, NZBINS, NZBINS) if ZBIN else np.linspace(0, 1, NZBINS)
    z = PlotCfg('z', z_binning,
                lambda x : x[2])

    data.plot(figsdir, x.values() + [nn, nn_noA, z], 100000, order=range(2))

    cut_nn = nn.hists[0].quantile(0.95)
    zcut = PlotCfg('zcut', z_binning,
                   lambda x : x[2],
                   cut_fn=lambda x : (model.predict(1, x[0])>cut_nn))
    cut_nnnoA = nn_noA.hists[0].quantile(0.95)
    zcut_noA = PlotCfg('zcut_noA', z_binning,
                       lambda x : x[2],
                       cut_fn=lambda x : (model_noA.predict(1, x[0])>cut_nnnoA))

    data.plot(figsdir, [zcut, zcut_noA], 100000, order=range(2))
