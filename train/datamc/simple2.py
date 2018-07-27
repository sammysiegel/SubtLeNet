#!/usr/bin/env python2.7

from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from subtlenet.backend.keras_objects import *
from subtlenet.backend.losses import *
from subtlenet.backend.layers import *

import os, numpy as np
from sys import stdout, exit
from tqdm import tqdm
from subtlenet import utils

utils.set_processor('cpu')
sqr = np.square
D = 5
ZLO = 0
ZHI = 2
ZBINS = 20

def _make_parent(path):
    os.system('mkdir -p %s' % ('/').join(path.split('/')[:-1]))

def idx(l, data):
    return [data[i] for i in l]


class PlotCfg(object):
    def __init__(self, name, binning, fn, classes=[0, 1], 
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
        for _, h in self.hists.iteritems():
            h.clear()

    def add_data(self, data):
        x = self._fn(data)
        weight = self._weight_fn(data) if self._weight_fn is not None else None
        cut    = self._cut_fn(data)    if self._cut_fn is not None    else np.ones(x.shape[0]).astype(bool)
        for c in self.classes:
            mask = np.logical_and(cut, data[Data.YIDX] == c)
            w = weight[mask] if weight is not None else None
            self.hists[c].fill_array(x[mask], weights=w)


class Data(object):
    XIDX = 0
    ZIDX = 1
    YIDX = 2
    WIDX = 3
    MIDX = 4
    UIDX = 5

    def __init__(self, n_input=2, n_output=1):
        self.n_input = n_input
        self.n_output = n_output
        self.plotter = utils.Plotter()
        self.roc = utils.Roccer()

    def gen(self, N, bin=True):
        s_x = np.random.randn(N, self.n_input) * 0.5 + 1
        b_x_g = np.random.randn(N, self.n_input - 1) * 0.5 + 0.25
        b_x_e = np.random.exponential(size=(N, 1))
        b_x = np.concatenate([b_x_g, b_x_e], axis=-1)
        x = np.concatenate([s_x, b_x], axis=0)
        y = np.concatenate([np.ones(N), np.zeros(N)], axis=0)
        if bin:
            y = np_utils.to_categorical(y, 2)
        z = x[:, -1]
        mask = np.logical_and(z > ZLO, z < ZHI)
        z = (z[mask] - ZLO) / (ZHI - ZLO)
        x = x[mask]
        y = y[mask]
        w = np.ones(y.shape[0])
        if bin:
            w_masked = np.copy(y[:, 0])
        else:
            w_masked = 1 - y
        idx = np.random.permutation(y.shape[0])
        return (x[idx], z[idx], y[idx], 
                w[idx], w_masked[idx], 
                np.random.uniform(size=y.shape[0]))

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
            for i, label in enumerate(order):
                if label not in cfg.hists:
                    continue
                if scale:
                    cfg.hists[label].scale()
                self.plotter.add_hist(cfg.hists[label], str(label), i)

            self.plotter.plot(xlabel=cfg.xlabel, ylabel=cfg.ylabel, output=outpath + '/' + cfg.name)

        return


class DModel(object):

    def __init__(self, n_inputs=2, penalty=0):
        losses = [categorical_crossentropy]
        losses += [pred_loss] * 2
        losses += [kernel_loss]

        X = Input(shape=(n_inputs,), name='X')
        h = BatchNormalization()(X)
        h = Dense(2 * n_inputs, activation='tanh', kernel_initializer='lecun_uniform')(h)
        h = BatchNormalization()(h)
        h = Dense(2 * n_inputs, activation='tanh', kernel_initializer='lecun_uniform')(h)
        h = BatchNormalization()(h)
        h = Dense(n_inputs, activation='relu', kernel_initializer='lecun_uniform')(h)
        h = BatchNormalization()(h)
        h = Dense(n_inputs, activation='relu', kernel_initializer='lecun_uniform')(h)
        h = BatchNormalization()(h)
        Y = Dense(2, activation='softmax', kernel_initializer='lecun_uniform', name='Y')(h)

        self.discriminator = Model(inputs=[X], outputs=[Y])
        self.discriminator.compile(loss=losses[0], optimizer=Adam())
        DModel._sum('discriminator', self.discriminator)

        p0 = ConvexPolyLayer(D, alpha=0, name='poly0')
        p1 = ConvexPolyLayer(D, alpha=0, name='poly1')

        Z = Input(shape=(1, ), name='Z')
        Q0 = p0(Z)
        Q1 = p1(Z)

        Q1 = multiply([Q1, Lambda(lambda x : x[:, 1])(Y)])
        
        K0 = WeightLayer(p0)(Z)
        K1 = WeightLayer(p1)(Z)
        KDiff = concatenate([K0, K1], axis=1)
        print K.int_shape(KDiff)
        print K.int_shape(K1)

        # self.model = Model(inputs=[X, Z], outputs=[Y, Q0, Q1, KDiff])
        # self.model.compile(loss=losses, 
        #                    loss_weights=[1, 100 * penalty, 100 * penalty, penalty], 
        #                    optimizer=Adam())
        self.model = Model(inputs=[X, Z], outputs=[Y, Q0])
        utils.freeze(self.model, False, self.discriminator.layers) # unfreeze all of model except D
#        utils.freeze(self.discriminator, True)
        self.model.compile(loss=losses[:2], 
#                           loss_weights=[1, 100 * penalty, 100 * penalty, penalty], 
                           optimizer=Adam())
        DModel._sum('model', self.model)



    @staticmethod
    def _sum(name, model):
        s = '====== Model %s ======' % name
        print s
        model.summary()
        print '=' * len(s)

    def fit(self, model, x, y, w=None):
        model.fit(x, y, sample_weight=w, verbose=0)

    def tob(self, model, x, y, w=None):
        model.train_on_batch(x, y, sample_weight=w)

    def train(self, data_fn, batch_size, **kwargs):
        data = data_fn(batch_size)
        N = data[Data.XIDX].shape[0]
        '''
        x = [data[Data.XIDX], data[Data.ZIDX]]
        y = [data[Data.YIDX], data[Data.ZIDX], data[Data.ZIDX], np.zeros_like(data[Data.YIDX])]
        w = [data[Data.WIDX], data[Data.WIDX], data[Data.MIDX], data[Data.WIDX]]
        '''
        x = idx([Data.XIDX, Data.ZIDX], data)
        y = idx([Data.YIDX, Data.ZIDX], data) #+ [np.zeros(N*2*D*D).reshape(N,2,D*D)]
        w = idx([Data.WIDX, Data.MIDX], data) #+ [np.ones(N)]
        self.tob(self.model, x, y, w, **kwargs)

    def train_d(self, data_fn, batch_size, **kwargs):
        data = data_fn(batch_size)
        x = idx([Data.XIDX], data)
        y = idx([Data.YIDX], data)
        w = idx([Data.WIDX], data)
        self.tob(self.discriminator, x, y, w, **kwargs)

    def save_as_keras(self, path):
        _make_parent(path)
        self.DAs.save(path)
        print 'Saved to', path

    def save_as_tf(self, path):
        _make_parent(path)
        sess = K.get_session()
        print [ l.op.name for l in self.DAs.inputs ], '->', [ l.op.name for l in self.DAs.outputs ]
        graph = graph_util.convert_variables_to_constants(sess, 
                                                          sess.graph.as_graph_def(), 
                                                          [ n.op.name for n in self.DAs.outputs ])
        p0 = ('/').join(path.split('/')[:-1])
        p1 = path.split('/')[-1]
        graph_io.write_graph(graph, p0, p1, as_text=False)
        print 'Saved to', path

    def predict(self, cls, *args, **kwargs):
        return self.model.predict(*args, **kwargs)[0][:, cls]

    def get_poly(self, name):
        l = None
        for l_ in self.model.layers:
            if l_.name == name:
                l = l_ 
                break
        if l is None:
            return 
        coeffs = [x[0] for x in l.poly_coeffs]
        integral = l.integral
        poly_fn = np.polynomial.polynomial.Polynomial(coeffs)
        return lambda x, fn=poly_fn, i=integral : fn(x) / i



if __name__ == '__main__':
    figsdir = '/home/snarayan/public_html/figs/poly/v0/'

    model = DModel(penalty=100)
    model_noA = DModel()
    data = Data()

    batch = 128
    NBATCH = 5000
    for _ in tqdm(range(NBATCH)):
        model.train_d(data.gen, batch)
#        model_noA.train_d(data.gen, batch)
    for _ in tqdm(range(NBATCH)):
        model.train(data.gen, batch)
#        model_noA.train(data.gen, batch)
    #for _ in tqdm(range(NBATCH)):
    #    model.train_d(data.gen, batch)
    #    model_noA.train_d(data.gen, batch)

    xs = {i : PlotCfg('x%i'%i, 
                      np.linspace(-1, 3, 40), 
                      lambda x, i=i: x[Data.XIDX][:, i]) 
          for i in xrange(2)}
    nn = PlotCfg('nn', 
                 np.linspace(0, 1, 40), 
                 lambda x: model.discriminator.predict(x[Data.XIDX])[:,1])
    nn_noA = PlotCfg('nn_noA', 
                      np.linspace(0, 1, 40), 
                      lambda x: model_noA.discriminator.predict(x[Data.XIDX])[:,1])

    z_binning = np.linspace(0, 1, 40)
    z = PlotCfg('z', z_binning, lambda x: x[Data.ZIDX])
    zcut = PlotCfg('zcut', z_binning, lambda x: x[Data.ZIDX],
                    cut_fn=lambda x : model.discriminator.predict(x[Data.XIDX])[:,1]>0.8)
    zw = PlotCfg('zw', z_binning, lambda x: x[Data.ZIDX], 
                 weight_fn=lambda x : model.discriminator.predict(x[Data.XIDX])[:,1])

    Q0 = model.get_poly('poly0')
    q0 = PlotCfg('q0',
                 z_binning,
                 lambda x : x[Data.UIDX],
                 weight_fn=lambda x : Q0(x[Data.UIDX]))
    #Q1 = model.get_poly('poly1')
    #q1 = PlotCfg('q1',
    #             z_binning,
    #             lambda x : x[Data.UIDX],
    #             weight_fn=lambda x : Q1(x[Data.UIDX]))

    data.plot(figsdir, [nn, q0,  z, zw, zcut] + xs.values(), 100000, order=range(2))
