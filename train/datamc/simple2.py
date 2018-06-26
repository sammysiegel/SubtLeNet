from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from subtlenet.backend.keras_objects import *
from subtlenet.backend.losses import *
from subtlenet.backend.layers import *

import os, numpy as np
from sys import stdout
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
            mask = np.logical_and(cut, data[1] == c)
            w = weight[mask] if weight is not None else None
            self.hists[c].fill_array(x[mask], weights=w)

        return


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
        return x[idx], z[idx], y[idx], w[idx], w_masked[idx], np.random.uniform(y.shape[0])

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
        h = Dense(2 * n_inputs, activation='tanh', kernel_initializer='lecun_uniform')(X)
        h = Dense(2 * n_inputs, activation='relu', kernel_initializer='lecun_uniform')(h)
        h = Dense(n_inputs, activation='relu', kernel_initializer='lecun_uniform')(h)
        h = Dense(n_inputs, activation='relu', kernel_initializer='lecun_uniform')(h)
        Y = Dense(2, activation='softmax', kernel_initializer='lecun_uniform', name='Y')(h)

        p0 = ConvexPolyLayer(D, alpha=0, name='poly0')
        p1 = ConvexPolyLayer(D, alpha=0, name='poly1')

        Z = Input(shape=(1, ), name='Z')
        Q0 = p0(Z, return_coeffs=False)
        K0 = p0(Z, return_coeffs=True)

        P1 = p1(Z, return_coeffs=False)
        prob1 = Lambda(lambda x: x[:, 1])(Y)
        Q1 = multiply([prob1, P1])
        K1 = p1(Z, return_coeffs=True)
        K = concatenate([K0, K1], axis=0)

        # self.model = Model(inputs=[X, Z], outputs=[Y, Q0, Q1, K])
        # self.model.compile(loss=losses, 
        #                    loss_weights=[1, 100 * penalty, 100 * penalty, penalty], 
        #                    optimizer=Adam())
        self.model = Model(inputs=[Z], outputs=[Q0])
        self.model.compile(loss=losses[1:2], 
                           optimizer=Adam())
        DModel._sum('model', self.model)

    @staticmethod
    def _sum(name, model):
        s = '====== Model %s ======' % name
        print s
        model.summary()
        print '=' * len(s)

    def fit(self, x, y, w=None):
        self.model.fit(x, y, sample_weight=w, verbose=0)

    def tob(self, x, y, w=None):
        self.model.train_on_batch(x, y, sample_weight=w)

    def train(self, data_fn, batch_size, **kwargs):
        data = data_fn(batch_size)
        x = [data[Data.XIDX], data[Data.ZIDX]]
        y = [data[Data.YIDX], data[Data.ZIDX], data[Data.ZIDX], np.zeros_like(data[Data.YIDX])]
        w = [data[Data.WIDX], data[Data.WIDX], data[Data.MIDX], data[Data.WIDX]]
        self.tob(x, y, w, **kwargs)

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

    data = Data()
    model = DModel(penalty=10)
    model_noA = DModel()

    batch = 128
    for _ in tqdm(range(500)):
        model.train(data.gen, batch)
        model_noA.train(data.gen, batch)

    xs = {i : PlotCfg('x%i'%i, 
                      np.linspace(-1, 3, 40), 
                      lambda x, i=i: x[0][:, i]) 
          for i in xrange(2)}
    nn = PlotCfg('nn', 
                 np.linspace(0, 1, 40), 
                 lambda x: model.predict(1, [x[Data.XIDX], x[Data.ZIDX]]))
    nn_noA = PlotCfg('nn_noA', 
                      np.linspace(0, 1, 40), 
                      lambda x: model_noA.predict(1, [x[Data.XIDX], x[Data.ZIDX]]))

    z_binning = np.linspace(0, 1, ZBINS)
    z = PlotCfg('z', z_binning, lambda x: x[2])

    Q0 = model.get_poly('poly0')
    q0 = PlotCfg('q0',
                 z_binning,
                 lambda x : x[Data.UIDX],
                 weight_fn=Q0(x[Data.UIDX]))

    data.plot(figsdir, [q0] + xs.values(), 100000, order=range(2))
