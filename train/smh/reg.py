#!/usr/bin/env python

from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from subtlenet.backend.keras_objects import *
from subtlenet.backend.losses import *
from tensorflow.python.framework import graph_util, graph_io
from glob import glob

from subtlenet import utils
utils.set_processor('cpu')

import os

def _make_parent(path):
    os.system('mkdir -p %s'%('/'.join(path.split('/')[:-1])))

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
        for fn_name,f in fns.iteritems():
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
    def load(self, idx):
        print self._files[idx],
        f = np.load(self._files[idx])
        print f['shape']
        return f
    def get_shape(self, key):
        f = self.load(0)
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
            data = dict(self.load(idx))
            data[name] = f(data)
            np.savez(fpath, **data)
    def plot(self, outpath, cfgs):
        outpath += '/'
        _make_path(outpath)
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
        h = self._Dense(h)
        h = LeakyReLU(0.2)(h)
        h = self._Dense(h)
        outputs = [self._Dense(h,1) for _ in xrange(self.n_targets)]

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=Adam(),
                           loss=losses,
                           loss_weights=loss_weights)
    def _Dense(self, h, n=-1):
        if n < 0:
            n = 2 * self.n_inputs
        return Dense(n, activation='linear', kernel_initializer='lecun_uniform')(h)
    def train(self, data, steps_per_epoch=100, epochs=10, validation_steps=10, callbacks=None):
        self.model.fit_generator(data('train'),
                                 validation_data=data('val'),
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=epochs,
                                 validation_steps=validation_steps,
                                 callbacks=callbacks)
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
        self.model.predict(*args, **kwargs)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_arg('--train', action='store_true')
    parser.add_arg('--plot', action='store_true')
    parser.add_arg('--version', type=int, default=0)
    args = parser.parse_args()

    version = 'v%i'%(args.version)
    reader = Reader('/data/t3serv014/snarayan/deep/v_010_2/*npz',
                    keys={'input':['inputs'],
                          'target':['jotGenPt/jotPt','jotGenEta','jotGenPhi']})
    regmodel = RegModel(reader.get_shape('inputs')[1], 3)
    if args.train:
        regmodel.train(reader)
        regmodel.save_as_keras('models/'+version+'/weights.h5')
        regmodel.save_as_tf('models/'+version+'/graph.pb')
        reader.add_coll(version, lambda x : regmodel.predict(x['inputs']))
    if args.plot:
        pt = PlotCfg('pt', np.linspace(25, 100, 15),
                     {'Truth' : lambda x : x['jotGenPt'],
                      'Reco'  : lambda x : x['jotPt'],
                      'Corr'  : lambda x : x['jotPt'] * x['v0'][0]},
                     xlabel=r'$p_\mathrm{T} [GeV]$',
                     ylabel='Jets')
        eta = PlotCfg('pt', np.linspace(-5, 5, 15),
                     {'Truth' : lambda x : x['jotGenEta'],
                      'Reco'  : lambda x : x['jotEta'],
                      'Corr'  : lambda x : x['v0'][1]},
                     xlabel=r'$\eta$',
                     ylabel='Jets')
        phi = PlotCfg('pt', np.linspace(-5, 5, 15),
                     {'Truth' : lambda x : x['jotGenPhi'],
                      'Reco'  : lambda x : x['jotPhi'],
                      'Corr'  : lambda x : x['v0'][2]},
                     xlabel=r'$\phi$',
                     ylabel='Jets')
        reader.plot('/home/snarayan/public_html/figs/smh/v0/breg/', [pt, eta, phi])
