#!/usr/bin/env python

from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from subtlenet.backend.keras_objects import *
from subtlenet.backend.losses import *
from tensorflow.python.framework import graph_util, graph_io
from glob import glob
import os
import numpy as np
from collections import namedtuple
from sys import stdout

from subtlenet import utils
utils.set_processor('cpu')

def _make_parent(path):
    os.system('mkdir -p %s'%('/'.join(path.split('/')[:-1])))

def throw_toy(mu, down, up):
    sigma = np.abs(up - down) / 2
    return np.random.normal(mu, sigma)

def sqr(x):
    return np.square(x)

Vec = namedtuple('Vec', ['x','y','z','t'])
def convert4(pt, eta, phi, m):
    x = pt * np.cos(phi)
    y = pt * np.sin(phi)
    z = pt * np.sinh(eta)
    t = np.sqrt(sqr(pt * np.cosh(eta)) + sqr(m))
    return Vec(x,y,z,t)

def mjj(pt0, eta0, phi0, m0, pt1, eta1, phi1, m1):
    v0 = convert4(pt0, eta0, phi0, m0)
    v1 = convert4(pt1, eta1, phi1, m1)
    return np.sqrt(sqr(v0.t + v1.t)
                   - sqr(v0.x + v1.x)
                   - sqr(v0.y + v1.y)
                   - sqr(v0.z + v1.z))

class PlotCfg(object):
    def __init__(self, name, binning, fns, weight_fn=None, xlabel=None, ylabel=None):
        self._fns = fns
        self._weight_fn = weight_fn
        self.name = name
        self._bins = binning
        self.hists = {x:utils.NH1(binning) for x in fns}
        self.xlabel = xlabel
        self.ylabel = ylabel
    def clear(self):
        for _,h in self.hists.iteritems():
            h.clear()
    def add_data(self, data):
        weight = self._weight_fn(data) if self._weight_fn is not None else None
        for fn_name,f in self._fns.iteritems():
            h = self.hists[fn_name]
            x = f(data)
            h.fill_array(x, weights=weight)


class Reader(object):
    def __init__(self, path, keys, train_frac=0.6, val_frac=0.2):
        self._path = path
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
                    print
                    print 'Refreshing',self._path
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
                lo = hi; hi += batch_size
                yield r
    def add_coll(self, *args):
        if type(args[0]) == str:
            args = [tuple(args)]
        for idx,fpath in enumerate(self._files):
            stdout.write('%i/%i\r'%(idx, len(self._files))); stdout.flush()
            data = dict(self.load(idx))
            for name,f in args:
                data[name] = f(data)
            np.savez(fpath, **data)
        print
    def plot(self, outpath, cfgs, order=None):
        outpath += '/'
        _make_parent(outpath)
        self._idx['test'] = 0
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
            if order is None:
                order = cfg.hists.keys()
            for i,label in enumerate(order):
                if label not in cfg.hists:
                    continue
                self.plotter.add_hist(cfg.hists[label], label, i)
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
        inputs = Input(shape=(n_inputs,), name='input')
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
        outputs = [self._Dense(h, 1, 'linear', name='output_%i'%i) for i in xrange(self.n_targets)]

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=Adam(),
                           loss=losses,
                           loss_weights=loss_weights)
        self.model.summary()
    def _Dense(self, h, n=-1, activation='linear', name=None):
        if n < 0:
            n = 2 * self.n_inputs
        return Dense(n, activation=activation, kernel_initializer='lecun_uniform', name=name)(h)
    def train(self, data, steps_per_epoch=1000, epochs=5, validation_steps=50, callbacks=None):
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
        print 'Saved to',path
    def save_as_tf(self,path):
        _make_parent(path)
        sess = K.get_session()
        print [l.op.name for l in self.model.inputs],'->',[l.op.name for l in self.model.outputs]
        graph = graph_util.convert_variables_to_constants(sess,
                                                          sess.graph.as_graph_def(),
                                                          [n.op.name for n in self.model.outputs])
        p0 = '/'.join(path.split('/')[:-1])
        p1 = path.split('/')[-1]
        graph_io.write_graph(graph, p0, p1, as_text=False)
        print 'Saved to',path
    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)


if __name__ == '__main__':
    datadir = '/fastscratch/snarayan/breg/v_010_3/'
    figsdir = '/home/snarayan/public_html/figs/smh/v4/compare/breg/'
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--version', type=int, default=4)
    parser.add_argument('--quantiles', action='store_true')
    parser.add_argument('--inputs', type=str, default='inputs')
    args = parser.parse_args()

    inputs = args.inputs
    target = ['jotGenPt/jotPt']
    losses = [huber]
    version = 'v%i'%(args.version)
    if args.quantiles:
        target += ['jotGenPt/jotPt', 'jotGenPt/jotPt']
        losses += [QL(0.15), QL(0.85)]
        target += ['jotGenDEta'] * 3
        losses += [huber, QL(0.15), QL(0.85)]
        target += ['jotGenDPhi'] * 3
        losses += [huber, QL(0.15), QL(0.85)]
        target += ['jotGenM/jotM'] * 3
        losses += [huber, QL(0.15), QL(0.85)]
        version += '/quantiles'
    else:
        target += ['jotGenDEta', 'jotGenDPhi']
        losses += [huber, huber]
        version += '/means'
    version += inputs.replace('inputs','')

    reader = Reader(datadir+'/T*npz',
                    keys={'input':[inputs],
                          'target':target})
    reader_h = Reader(datadir+'Z*npz', keys={})
    reader_w = Reader(datadir+'W*npz', keys={})

    if args.train:
        regmodel = RegModel(reader.get_shape(inputs)[1], len(reader.get_target()),
                            losses=losses)
        regmodel.train(reader)
        regmodel.save_as_keras('models/'+version+'/weights.h5')
        regmodel.save_as_tf('models/'+version+'/graph.pb')

        reader_h.add_coll((version+'_hbb0', lambda x : regmodel.predict(x[inputs+'_hbb0'])),
                          (version+'_hbb1', lambda x : regmodel.predict(x[inputs+'_hbb1'])))
        reader_w.add_coll((version+'_hbb0', lambda x : regmodel.predict(x[inputs+'_hbb0'])),
                          (version+'_hbb1', lambda x : regmodel.predict(x[inputs+'_hbb1'])))
        reader.add_coll(version, lambda x : regmodel.predict(x[inputs]))

    if args.plot:
        def pt_throw_toy(x, key='v4/quantiles'):
            return throw_toy(x[key][0].reshape(-1),
                             x[key][1].reshape(-1),
                             x[key][2].reshape(-1))
        def eta_throw_toy(x, key='v4/quantiles'):
            return throw_toy(x[key][3].reshape(-1),
                             x[key][4].reshape(-1),
                             x[key][5].reshape(-1))
        def phi_throw_toy(x, key='v4/quantiles'):
            return throw_toy(x[key][6].reshape(-1),
                             x[key][7].reshape(-1),
                             x[key][8].reshape(-1))
        def m_throw_toy(x, key='v4/quantiles'):
            return throw_toy(x[key][9].reshape(-1),
                             x[key][10].reshape(-1),
                             x[key][11].reshape(-1))

        eta = PlotCfg('eta', np.linspace(-0.2,0.2,50),
                      {'Truth' : lambda x : x['jotGenDEta'],
                       'Mean'  : lambda x : x['v4/quantiles'][3].reshape(-1),
                       'Norm'  : lambda x : eta_throw_toy(x),
                       },
                      xlabel=r'$\Delta \eta$', ylabel='Jets')
        phi = PlotCfg('phi', np.linspace(-0.2,0.2,50),
                      {'Truth' : lambda x : x['jotGenDPhi'],
                       'Mean'  : lambda x : x['v4/quantiles'][6].reshape(-1),
                       'Norm'  : lambda x : phi_throw_toy(x),
                       },
                      xlabel=r'$\Delta \phi$', ylabel='Jets')
        pt_ratio = PlotCfg('pt_ratio', np.linspace(0, 2.5, 50),
                     {'Truth' : lambda x : x['jotGenPt/jotPt'],
                      'Mean'  : lambda x : x['v4/quantiles'][0].reshape(-1),
                      'Norm'  : pt_throw_toy,
                      },
                     xlabel=r'$p_\mathrm{T}^\mathrm{truth}/p_\mathrm{T}^\mathrm{reco}$',
                     ylabel='Jets')
        pt = PlotCfg('pt', np.linspace(0, 200, 50),
                     {'Truth' : lambda x : x['jotGenPt'],
                      'Mean'  : lambda x : x['jotPt'] * x['v4/quantiles'][0].reshape(-1),
                      'Norm'  : lambda x : x['jotPt'] * pt_throw_toy(x),
                      'Reco'  : lambda x : x['jotPt'],
                      },
                     xlabel=r'$p_\mathrm{T}$ [GeV]',
                     ylabel='Jets')
        mass_ratio = PlotCfg('mass_ratio', np.linspace(0, 2.5, 50),
                     {'Truth' : lambda x : x['jotGenM/jotM'],
                      'Mean'  : lambda x : x['v4/quantiles'][9].reshape(-1),
                      'Norm'  : m_throw_toy,
                      },
                     xlabel=r'$m^\mathrm{truth}/m^\mathrm{reco}$',
                     ylabel='Jets')
        mass = PlotCfg('mass', np.linspace(0, 50, 50),
                     {'Truth' : lambda x : x['jotGenM'],
                      'Mean'  : lambda x : x['jotM'] * x['v4/quantiles'][9].reshape(-1),
                      'Norm'  : lambda x : x['jotM'] * m_throw_toy(x),
                      'Reco'  : lambda x : x['jotM'],
                      },
                     xlabel=r'$p_\mathrm{T}$ [GeV]',
                     ylabel='Jets')
        error = PlotCfg('error', np.linspace(0, 2.5, 50),
                     {'Mean'  : lambda x : np.abs(x['v4/quantiles'][0].reshape(-1) - x['jotGenPt/jotPt']),
                      'Norm'  : lambda x : np.abs(pt_throw_toy(x) - x['jotGenPt/jotPt']),
                      },
                     xlabel=r'$|y-\hat{y}|$',
                     ylabel='Jets')
        normerror = PlotCfg('normerror', np.linspace(0, 2.5, 50),
                     {'Mean'  : lambda x : np.abs(x['v4/quantiles'][0].reshape(-1) - x['jotGenPt/jotPt']),
                      'Norm'  : lambda x : np.divide(
                                        np.abs(pt_throw_toy(x) - x['jotGenPt/jotPt']),
                                        (x['v4/quantiles'][2].reshape(-1) - x['v4/quantiles'][1].reshape(-1))
                                    ),
                      },
                     xlabel=r'$|y-\hat{y}| / (q_{0.85}-q_{0.15})$',
                     ylabel='Jets')
        order = ['Truth', 'Reco', 'Mean', 'Norm', r'Norm $\Delta R$', r'Norm $\Delta \eta\phi$']
#        reader.plot(figsdir,
#                    [eta, phi, normerror, error, pt_ratio, pt, mass, mass_ratio],
#                    order=order)

        def get_hbbm(x, scales=((1,0,0,1),(1,0,0,1))):
            args = []
            for i in [0,1]:
                pt = x['jotPt[hbbjtidx[%i]]'%i] * scales[i][0]
                eta = x['jotEta[hbbjtidx[%i]]'%i] + scales[i][1]
                phi = x['jotPhi[hbbjtidx[%i]]'%i] + scales[i][2]
                m = x['jotM[hbbjtidx[%i]]'%i] * scales[i][3]
                args.extend([pt,eta,phi,m])
            return mjj(*args)

        def get_genhbbm(x):
            args = []
            for i in [0,1]:
                pt = x['jotGenPt[hbbjtidx[%i]]'%i]
                m = x['jotGenM[hbbjtidx[%i]]'%i]
                eta = x['jotGenEta[hbbjtidx[%i]]'%i]
                phi = x['jotGenPhi[hbbjtidx[%i]]'%i]
                args.extend([pt,eta,phi,m])
            return mjj(*args)

        hbbm = PlotCfg('hbbm', np.linspace(0, 200, 50),
                       {
                          'Truth' : lambda x : get_genhbbm(x),
                          'Reco' : lambda x : get_hbbm(x),
                          'Mean' : lambda x : get_hbbm(x, ((x['v4/quantiles_hbb0'][0].reshape(-1),
                                                            x['v4/quantiles_hbb0'][3].reshape(-1),
                                                            x['v4/quantiles_hbb0'][6].reshape(-1),
                                                            x['v4/quantiles_hbb0'][9].reshape(-1),
                                                           ),
                                                           (x['v4/quantiles_hbb1'][0].reshape(-1),
                                                            x['v4/quantiles_hbb1'][3].reshape(-1),   
                                                            x['v4/quantiles_hbb1'][6].reshape(-1),
                                                            x['v4/quantiles_hbb1'][9].reshape(-1),
                                                           ))),
                          'Test' : lambda x : get_hbbm(x, ((x['v4/quantiles_hbb0'][0].reshape(-1),
                                                            x['v4/quantiles_hbb0'][3].reshape(-1),
                                                            x['v4/quantiles_hbb0'][6].reshape(-1),
                                                            m_throw_toy(x, 'v4/quantiles_hbb0'),
                                                           ),
                                                           (x['v4/quantiles_hbb1'][0].reshape(-1),
                                                            x['v4/quantiles_hbb1'][3].reshape(-1),   
                                                            x['v4/quantiles_hbb1'][6].reshape(-1),
                                                            m_throw_toy(x, 'v4/quantiles_hbb1'),
                                                           ))),
                          'Norm' : lambda x : get_hbbm(x, ((pt_throw_toy(x, 'v4/quantiles_hbb0'),
                                                            eta_throw_toy(x, 'v4/quantiles_hbb0'),
                                                            phi_throw_toy(x, 'v4/quantiles_hbb0'),
                                                            m_throw_toy(x, 'v4/quantiles_hbb0')),
                                                           (pt_throw_toy(x, 'v4/quantiles_hbb1'),
                                                            eta_throw_toy(x, 'v4/quantiles_hbb1'),
                                                            phi_throw_toy(x, 'v4/quantiles_hbb1'),
                                                            m_throw_toy(x, 'v4/quantiles_hbb1')))),
                       },
                      xlabel=r'$m_H$ [GeV]', ylabel='Events')
        pt_ratio = PlotCfg('pt_ratio_hbb0', np.linspace(0, 2.5, 50),
                     {'Truth' : lambda x : x['jotGenPt[hbbjtidx[0]]/jotPt[hbbjtidx[0]]'],
                      'Mean'  : lambda x : x['v4/quantiles_hbb0'][0].reshape(-1),
                      'Norm'  : lambda x : pt_throw_toy(x, 'v4/quantiles_hbb0'),
                      },
                     xlabel=r'$p_\mathrm{T}^\mathrm{truth}/p_\mathrm{T}^\mathrm{reco}$',
                     ylabel='Jets')
        pt = PlotCfg('pt_hbb0', np.linspace(0, 200, 50),
                     {'Truth' : lambda x : x['jotGenPt[hbbjtidx[0]]'],
                      'Mean'  : lambda x : x['jotPt[hbbjtidx[0]]'] * x['v4/quantiles_hbb0'][0].reshape(-1),
                      'Norm'  : lambda x : x['jotPt[hbbjtidx[0]]']
                                            * pt_throw_toy(x, 'v4/quantiles_hbb0'),
                      'Reco'  : lambda x : x['jotPt[hbbjtidx[0]]'],
                      },
                     xlabel=r'$p_\mathrm{T}$ [GeV]',
                     ylabel='Jets')
        order = ['Truth', 'Reco', 'Mean', 'Norm', 'Test']
        reader_h.plot(figsdir,
                      [pt, pt_ratio, hbbm],
                      order = order)
        for cfg in [pt, pt_ratio, hbbm]:
            cfg.name += '_bkg'
            cfg.clear()
        reader_w.plot(figsdir,
                      [pt, pt_ratio, hbbm],
                      order = order)
