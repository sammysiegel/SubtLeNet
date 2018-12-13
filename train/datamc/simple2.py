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
D = 10
ZLO = 0
ZHI = 2
ZBINS = 20
THRESHOLD = 0.6
#THRESHOLDS = np.linspace(0.5, 0.9, 5)
THRESHOLDS = [0.2, 0.5, 0.8]
SCALE = 100

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
        self.roc = utils.Roccer(y_range=range(-1,1))

    def gen(self, N, bin=True):
        s_x_0 = np.random.randn(N, self.n_input - 1) * 0.5 + 1
        s_x_1 = np.random.randn(N, 1) * 0.5 + 1
        s_x = np.concatenate([s_x_0, s_x_1], axis=-1)
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

    def plot(self, outpath, cfgs, N, order=None, scale=True, suffix=''):
        outpath += '/'
        _make_parent(outpath)
        _batch_size = 50000
        for _ in tqdm(range(N / _batch_size)):
            for cfg in cfgs:
                cfg.add_data(self.gen(_batch_size, bin=False))

        hists = {}
        for cfg in cfgs:
            self.plotter.clear()
            if order is None:
                order = cfg.hists.keys()
            for i, label in enumerate(order):
                if label not in hists:
                    hists[label] = {}
                if label not in cfg.hists:
                    continue
                if scale:
                    cfg.hists[label].scale()
                self.plotter.add_hist(cfg.hists[label], str(label), i)
                hists[label][cfg.name] = cfg.hists[label]

            self.plotter.plot(xlabel=cfg.xlabel, ylabel=cfg.ylabel, output=outpath + '/' + cfg.name + suffix)

        to_plot=['nn','x0','x1']

        self.roc.clear()
        self.roc.add_vars(hists[1], hists[0], {x:x for x in to_plot})
        self.roc.plot(outpath + '/roc_' + suffix) 

        return



class DModel(object):
    def __init__(self, n_inputs=2, loss_weights=[1]):
        self.loss_weights = loss_weights

        X = Input(shape=(n_inputs,), name='X')
        h = X
        h = BatchNormalization()(h)
        h = Dense(2 * n_inputs, activation='tanh', kernel_initializer='lecun_uniform')(h)
        h = Dense(2 * n_inputs, activation='tanh', kernel_initializer='lecun_uniform')(h)
        Y = Dense(2, activation='softmax', kernel_initializer='lecun_uniform', name='Y')(h)

        self.discriminator = Model(inputs=[X], outputs=[Y])

        p0 = ConvexPolyLayer(D, alpha=0.01, name='poly0')

        Z = Input(shape=(1, ), name='Z')
        Q0 = p0(Z)
        k0 = WeightLayer(p0, name='k0')
        K0 = k0(Q0)

        self.Ks = []
        self.Qs = []

        def add_bin(lo, hi=None):
            if not hi:
                hi = lo + 0.1 
            name = DModel.bname(lo)

            Yhat = Lambda(
    #                    lambda y : K.expand_dims(y[:, 1], axis=-1), 
    #                    lambda y : K.expand_dims(K.cast(y[:, 1] > THRESHOLD, 'float32')), 
                        lambda y, lo=lo : K.expand_dims(K.sigmoid((y[:, 1] - lo) * SCALE )), 
                        name='yhat'+name
                    )(Y)
            YZ = ExpandAndConcat(name='eandc'+name)([Z, Yhat])
            p1 = ConvexPolyLayer(D, alpha=0.01, name='poly1'+name, weighted=True)
            Q1 = p1(YZ)

            k1 = WeightLayer(p1, name='k1'+name)
            K1 = k1(Q1)
            KDiff = concatenate([K0, K1], axis=1)

            self.Qs.append(Q1)
            self.Ks.append(KDiff)

            self.loss_weights += [0,0]

        for i in THRESHOLDS:
            add_bin(i)
        
        self.frozen_model = Model(inputs=[X, Z], outputs=[Y, Q0]+self.Qs)
        self.model = Model(inputs=[X, Z], outputs=[Y]+self.Qs+self.Ks)

        utils.freeze(self.discriminator, False)
        self.losses = [categorical_crossentropy, pred_loss, kernel_loss]
        losses = [cce, pred_loss] + ([pred_loss] * len(self.Qs))
        self.frozen_model.compile(loss=losses, optimizer=Adam())

        utils.freeze(self.discriminator, True)
        self.discriminator.compile(loss=cce, optimizer=Adam())

        k0.trainable = False
        p0.trainable = False
        self.recompile_model() # compile self.model for the first time 

        DModel._sum('discriminator', self.discriminator)
        DModel._sum('frozen_model', self.frozen_model)
        DModel._sum('model', self.model)

    @staticmethod
    def bname(lo):
        return '%.2i'%(int(lo*100))

    def recompile_model(self, loss_weights=None):
        if loss_weights is not None:
            self.loss_weights = loss_weights 
        loss = [cce]
        loss += ([pred_loss] * len(self.Qs))
        loss += ([kernel_loss] * len(self.Ks))
        self.model.compile(loss=loss, 
                           loss_weights=self.loss_weights, 
                           optimizer=Adam())
    @staticmethod
    def _sum(name, model):
        s = '====== Model %s ======' % name
        print s
        model.summary()
        print '=' * len(s)

    def fit(self, model, x, y, w=None):
        model.fit(x, y, sample_weight=w, verbose=0)

    def tob(self, model, x, y, w=None):
        return model.train_on_batch(x, y, sample_weight=w)

    def train_model(self, data_fn, batch_size, **kwargs):
        nQ = len(self.Qs)
        data = data_fn(batch_size)
        N = data[Data.XIDX].shape[0]
        x = idx([Data.XIDX, Data.ZIDX], data)
        y = idx([Data.YIDX]+([Data.ZIDX]*nQ), data) + ([np.zeros(N*2*D*D).reshape(N,2,D*D)]*nQ)
        w = idx([Data.WIDX]+([Data.MIDX]*nQ), data) + ([np.ones(N)]*nQ)
        return self.tob(self.model, x, y, w, **kwargs)

    def train_frozen(self, data_fn, batch_size, **kwargs):
        nQ = len(self.Qs)
        data = data_fn(batch_size)
        N = data[Data.XIDX].shape[0]
        x = idx([Data.XIDX, Data.ZIDX], data)
        y = idx([Data.YIDX]+([Data.ZIDX]*(nQ+1)), data) 
        w = idx([Data.WIDX]+([Data.MIDX]*(nQ+1)), data) 
        return self.tob(self.frozen_model, x, y, w, **kwargs)

    def train_d(self, data_fn, batch_size, **kwargs):
        data = data_fn(batch_size)
        x = idx([Data.XIDX], data)
        y = idx([Data.YIDX], data)
        w = idx([Data.WIDX], data)
        return self.tob(self.discriminator, x, y, w, **kwargs)

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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__ == '__main__':
    figsdir = '/home/snarayan/public_html/figs/poly/v0/'

    model = DModel()
    plot_model(model.model, to_file=figsdir+'/model.png')
    data = Data()

    batch = 128
    NBATCH = 5000
    
    xs = {i : PlotCfg('x%i'%i, 
                      np.linspace(-1, 3, 40), 
                      lambda x, i=i: x[Data.XIDX][:, i]) 
          for i in xrange(2)}
    nn = PlotCfg('nn', 
                 np.linspace(0, 1, 40), 
                 lambda x: model.discriminator.predict(x[Data.XIDX])[:,1])

    z_binning = np.linspace(0, 1, 40)
    z = PlotCfg('z', z_binning, lambda x: x[Data.ZIDX])
    zcuts = []
    qs = []
    for lo in THRESHOLDS:
        name = DModel.bname(lo) 
        zcuts.append( PlotCfg('zcut'+name, z_binning, lambda x: x[Data.ZIDX],
                              weight_fn=lambda x, lo=lo : sigmoid((model.discriminator.predict(x[Data.XIDX])[:,1] - lo) * SCALE)) )
        qs.append(PlotCfg('q1'+name,
                          z_binning,
                          lambda x : x[Data.UIDX],
                          weight_fn=lambda x, lo=lo : Qs[lo](x[Data.UIDX])))
    zw = PlotCfg('zw', z_binning, lambda x: x[Data.ZIDX], 
                 weight_fn=lambda x : np.power(model.discriminator.predict(x[Data.XIDX])[:,1], 2))

    q0 = PlotCfg('q0',
                 z_binning,
                 lambda x : x[Data.UIDX],
                 weight_fn=lambda x : Q0(x[Data.UIDX]))


    # for _ in tqdm(range(NBATCH * 2)):
    #     model.train_d(data.gen, batch * 5)
    for _ in tqdm(range(NBATCH)):
        model.train_frozen(data.gen, batch)
    print model.train_model(data.gen, batch)
    Q0 = model.get_poly('poly0')
    Qs = {lo:model.get_poly('poly1'+DModel.bname(lo)) for lo in THRESHOLDS}
#    data.plot(figsdir, [nn, q0,  z, zw] + zcuts + qs + xs.values(), 100000, order=range(2), suffix='_pre')


    model.recompile_model(loss_weights=[1,
                                        0.25, 0.25, 0.5,
                                        0.25, 0.25, 0.5])

    for _ in tqdm(range(NBATCH)):
        l = model.train_model(data.gen, batch * 2)
    print l 
    Q0 = model.get_poly('poly0')
    Qs = {lo:model.get_poly('poly1'+DModel.bname(lo)) for lo in THRESHOLDS}
    data.plot(figsdir, [nn, q0,  z, zw] + zcuts + qs + xs.values(), 100000, order=range(2), suffix='_post')
