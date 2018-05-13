#!/usr/bin/env python

from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from subtlenet.backend.keras_objects import *
from subtlenet.backend.losses import *
from tensorflow.python.framework import graph_util, graph_io
from glob import glob
import os
import numpy as np

from subtlenet import utils
# utils.set_processor('cpu')

def _make_parent(path):
    os.system('mkdir -p %s'%('/'.join(path.split('/')[:-1])))

def throw_toy(mu, down, up):
    sigma = (up - down) / 2
    return np.random.normal(mu, sigma)


class PlotCfg(object):
    def __init__(self, name, binning, fns, weight_fn=None, xlabel=None, ylabel=None):
        self._fns = fns
        self._weight_fn = weight_fn
        self.name = name
        self._bins = binning
        self.hists = {x:utils.NH1(binning) for x in fns}
        self.xlabel = xlabel
        self.ylabel = ylabel
    def add_data(self, data):
        weight = self._weight_fn(data) if self._weight_fn is not None else None
        for fn_name,f in self._fns.iteritems():
            h = self.hists[fn_name]
            x = f(data)
            h.fill_array(x, weights=weight)


class Reader(object):
    def __init__(self, path, keys, train_frac=0.6, val_frac=0.2):
        self._files = glob(path)
        self._idx = {'train':0, 'val':0, 'test':0}
        self.keys = keys
        self._fracs = {'train':(0,train_frac),
                       'val':(train_frac,train_frac+val_frac),
                       'test':(train_frac+val_frac,1)}
        self.plotter = utils.Plotter()
    def get_target(self):
        return self.keys['target']
    def load(self, idx):
        # print self._files[idx],
        f = np.load(self._files[idx])
        # print f['shape']
        return f
    def get_shape(self, key='shape'):
        f = self.load(0)
        if key == 'shape':
            return f[key]
        else:
            return f[key].shape
    def _gen(self, p, refresh):
        while True:
            if self._idx[p] == len(self._files):
                if refresh:
                    self._idx[p] = 0
                else:
                    return
            self._idx[p] += 1
            yield self.load(self._idx[p] - 1)
    def __call__(self, p, batch_size=1000, refresh=True):
        fracs = self._fracs[p]
        gen = self._gen(p, refresh)
        while True:
            f = next(gen)
            N = int(f['shape'][0] * fracs[1])
            lo = int(f['shape'][0] * fracs[0])
            hi = lo + batch_size if batch_size > 0 else N
            while hi <= N:
                r = [[f[x][lo:hi] for x in self.keys['input']]]
                r.append([f[x][lo:hi] for x in self.keys['target']])
                if 'weight' in self.keys:
                    r.append([f[x][lo:hi] for x in self.keys['weight']])
                # print [r[0][0][0], [r[1][x][0] for x in xrange(3)]]
                lo = hi; hi += batch_size
                yield r
    def add_coll(self, name, f):
        for idx,fpath in enumerate(self._files):
            print '%i/%i\r'%(idx, len(self._files)),
            data = dict(self.load(idx))
            data[name] = f(data)
            np.savez(fpath, **data)
        print
    def plot(self, outpath, cfgs):
        outpath += '/'
        _make_parent(outpath)
        gen = self._gen('test', refresh=False)
        try:
            while True:
                data = next(gen)
                for cfg in cfgs:
                    cfg.add_data(data)
        except StopIteration:
            pass
        for cfg in cfgs:
            self.plotter.clear()
            for i,(label,h) in enumerate(cfg.hists.iteritems()):
                self.plotter.add_hist(h, label, i)
            self.plotter.plot(xlabel=cfg.xlabel,
                              ylabel=cfg.ylabel,
                              output=outpath+'/'+cfg.name)
            self.plotter.plot(xlabel=cfg.xlabel,
                              ylabel=cfg.ylabel,
                              output=outpath+'/'+cfg.name+'_logy',
                              logy=True)


class RegModel(object):
    def __init__(self, n_inputs, n_targets, losses=None, loss_weights=None):
        if losses is None:
            losses = [huber] * n_targets

        self.n_inputs = n_inputs
        self.n_targets = n_targets
        inputs = Input(shape=(n_inputs,))
        h = inputs
        h = BatchNormalization(momentum=0.6)(h)
        h = self._Dense(h)
        h = LeakyReLU(0.2)(h)
        h = self._Dense(h)
        h = LeakyReLU(0.2)(h)
        h = self._Dense(h)
        h = LeakyReLU(0.2)(h)
        h = self._Dense(h)
        h = LeakyReLU(0.2)(h)
        h = self._Dense(h, activation='tanh')
        h = self._Dense(h)
        outputs = [self._Dense(h,1,'linear') for _ in xrange(self.n_targets)]

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=Adam(),
                           loss=losses,
                           loss_weights=loss_weights)
        self.model.summary()
    def _Dense(self, h, n=-1, activation='linear'):
        if n < 0:
            n = 2 * self.n_inputs
        return Dense(n, activation=activation, kernel_initializer='lecun_uniform')(h)
    def train(self, data, steps_per_epoch=1000, epochs=10, validation_steps=50, callbacks=None):
        history = self.model.fit_generator(data('train'),
                                           validation_data=data('val'),
                                           steps_per_epoch=steps_per_epoch,
                                           epochs=epochs,
                                           validation_steps=validation_steps,
                                           callbacks=callbacks)
        with open('history.log','w') as flog:
            history = history.history
            flog.write(','.join(history.keys())+'\n')
            for l in zip(*history.values()):
                flog.write(','.join([str(x) for x in l])+'\n')
    def save_as_keras(self, path):
        _make_parent(path)
        self.model.save(path)
    def save_as_tf(self,path):
        _make_parent(path)
        sess = K.get_session()
        graph = graph_util.convert_variables_to_constants(sess,
                                                          sess.graph.as_graph_def(),
                                                          [n.op.name for n in self.model.outputs])
        p0 = '/'.join(path.split('/')[:-1])
        p1 = path.split('/')[-1]
        graph_io.write_graph(graph, p0, p1, as_text=False)
    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--version', type=int, default=1)
    args = parser.parse_args()

    version = 'v%i'%(args.version)
    reader = Reader('/data/t3serv014/snarayan/deep/v_010_2/*npz',
                    keys={'input':['inputs'],
                         'target':[
                                   'jotGenPt/jotPt',
                                   'jotGenPt/jotPt', # for each quantile
                                   'jotGenPt/jotPt',
                                   # 'jotGenEta',
                                   # 'TMath::Sin(jotGenPhi)',
                                   # 'TMath::Cos(jotGenPhi)'
                                   ]})
    if args.train:
        regmodel = RegModel(reader.get_shape()[1], len(reader.get_target()),
                            losses=[huber, QL(0.15), QL(0.85)])
        regmodel.train(reader)
        reader.add_coll(version, lambda x : regmodel.predict(x['inputs']))
        regmodel.save_as_keras('models/'+version+'/weights.h5')
        regmodel.save_as_tf('models/'+version+'/graph.pb')
    if args.plot:
        pt_ratio = PlotCfg('pt_ratio', np.linspace(0, 2.5, 25),
                     {'Truth' : lambda x : x['jotGenPt/jotPt'],
                      'Corr 0'  : lambda x : x['v0'][0].reshape(-1),
                      'Corr 1'  : lambda x : throw_toy(x['v1'][0].reshape(-1),
                                                       x['v1'][1].reshape(-1),
                                                       x['v1'][2].reshape(-1)),
                      },
                     xlabel=r'$p_\mathrm{T}^\mathrm{truth}/p_\mathrm{T}^\mathrm{reco}$',
                     ylabel='Jets')
        pt = PlotCfg('pt', np.linspace(0, 200, 50),
                     {'Truth' : lambda x : x['jotGenPt'],
                      'Corr 0'  : lambda x : x['jotPt'] * x['v0'][0].reshape(-1),
                      'Corr 1'  : lambda x : x['jotPt'] *
                                             throw_toy(x['v1'][0].reshape(-1),
                                                       x['v1'][1].reshape(-1),
                                                       x['v1'][2].reshape(-1)),
                      'Reco'  : lambda x : x['jotPt'],
                      },
                     xlabel=r'$p_\mathrm{T} [GeV]$',
                     ylabel='Jets')
        eta = PlotCfg('eta', np.linspace(-2.5, 2.5, 25),
                     {'Truth' : lambda x : 2.5*x['jotGenEta'],
                      'Reco'  : lambda x : 2.5*x['jotEta'],
                      'Corr'  : lambda x : 2.5*x['v0'][1]},
                     xlabel=r'$\eta$',
                     ylabel='Jets')
        phi = PlotCfg('phi', np.linspace(-3.2, 3.2, 25),
                     {'Truth' : lambda x : 3.2*x['jotGenPhi'],
                      'Reco'  : lambda x : 3.2*x['jotPhi'],
                      'Corr'  : lambda x : np.arctan2(x['v0'][2],x['v0'][3])},
                     xlabel=r'$\phi$',
                     ylabel='Jets')
        reader.plot('/home/snarayan/public_html/figs/smh/v0/breg/', [pt_ratio, pt])
