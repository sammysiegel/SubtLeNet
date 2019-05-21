#!/usr/bin/env python

from sklearn.utils import shuffle
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
#from subtlenet.backend.keras_objects import *
#from subtlenet.backend.losses import *
from keras.layers import Dense, BatchNormalization, Input
from keras.utils import np_utils
from keras.optimizers import Adam
import keras.backend as K
from tensorflow.python.framework import graph_util, graph_io
import os, sys
import numpy as np
from collections import namedtuple

import subtlenet.utils as utils 
utils.set_processor('cpu')
VALSPLIT = 0.25 #0.7
MULTICLASS = False
REGRESSION = False

def _make_parent(path):
    os.system('mkdir -p %s'%('/'.join(path.split('/')[:-1])))

class Sample(object):
    def __init__(self, name, base, max_Y):
        self.name = name 
        self.X = np.load('%s/%s_%s.npy'%(base, name, 'x'))
        if REGRESSION:
            self.Y = np.load('%s/%s_%s.npy'%(base, name, 'y'))
        else:
            if MULTICLASS:
                self.Y = np_utils.to_categorical(
                            np.load('%s/%s_%s.npy'%(base, name, 'y')),
                            max_Y
                        )
            else:
                self.Y = np_utils.to_categorical(
                            (np.load('%s/%s_%s.npy'%(base, name, 'y')) > 0).astype(np.int),
                            2
                        )
        self.W = np.load('%s/%s_%s.npy'%(base, name, 'w'))
        self.idx = np.random.permutation(self.Y.shape[0])
    @property
    def tidx(self):
        if VALSPLIT == 1 or VALSPLIT == 0:
            return self.idx
        else:
            return self.idx[int(VALSPLIT*len(self.idx)):]
    @property
    def vidx(self):
        if VALSPLIT == 1 or VALSPLIT == 0:
            return self.idx
        else:
            return self.idx[:int(VALSPLIT*len(self.idx))]
    def infer(self, model):
        self.Yhat = model.predict(self.X)
    def standardize(self, mu, std):
        self.X = (self.X - mu) / std


class ClassModel(object):
    def __init__(self, n_inputs, n_hidden, n_targets):
        self._hidden = 0

        self.n_inputs = n_inputs
        self.n_targets = n_targets if MULTICLASS else 2
        self.n_hidden = n_hidden
        self.inputs = Input(shape=(n_inputs,), name='input')
        h = self.inputs
        h = BatchNormalization(momentum=0.6)(h)
        for _ in xrange(n_hidden-1):
            h = Dense(n_inputs, activation='relu')(h)
            h = BatchNormalization()(h)
        h = Dense(n_inputs, activation='tanh')(h)
        h = BatchNormalization()(h)
        if REGRESSION:
            self.outputs = Dense(1, activation='linear', name='output')(h)

            self.model = Model(inputs=self.inputs, outputs=self.outputs)
            self.model.compile(optimizer=Adam(),
                               loss='mse')
        else:
            self.outputs = Dense(self.n_targets, activation='softmax', name='output')(h)

            self.model = Model(inputs=self.inputs, outputs=self.outputs)
            self.model.compile(optimizer=Adam(),
                               loss='binary_crossentropy')
        self.model.summary()
    def train(self, samples):
        tX = np.vstack([s.X[s.tidx] for s in samples])
        tW = np.concatenate([s.W[s.tidx] for s in samples])
        vX = np.vstack([s.X[s.vidx] for s in samples])
        vW = np.concatenate([s.W[s.vidx] for s in samples])
        
        if REGRESSION:
            tY = np.concatenate([s.Y[s.tidx] for s in samples])
            vY = np.concatenate([s.Y[s.vidx] for s in samples])
        else:
            tY = np.vstack([s.Y[s.tidx] for s in samples])
            vY = np.vstack([s.Y[s.vidx] for s in samples])

        if not REGRESSION:
            for i in xrange(tY.shape[1]):
                tot = np.sum(tW[tY[:,i] == 1])
                tW[tY[:,i] == 1] *= 100/tot
                vW[vY[:,i] == 1] *= 100/tot

        history = self.model.fit(tX, tY, sample_weight=tW, 
                                 batch_size=1024, epochs=40, shuffle=True,
                                 validation_data=(vX, vY, vW))
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
    def load_model(self, path):
        self.model = load_model(path)


def plot(binning, fn, samples, outpath, xlabel=None, ylabel=None):
    hists = {}
    for s in samples:
        h = utils.NH1(binning)
        if type(fn) == int:
            h.fill_array(s.X[s.vidx,fn], weights=s.W[s.vidx])
        else:
            h.fill_array(fn(s), weights=s.W[s.vidx])
        h.scale()
        hists[s.name] = h

    p = utils.Plotter()
    for i,s in enumerate(samples):
        p.add_hist(hists[s.name], s.name, i)

    _make_parent(outpath)

    p.plot(xlabel=xlabel, ylabel=ylabel,
           output = outpath)
    p.plot(xlabel=xlabel, ylabel=ylabel,
           output = outpath + '_logy',
           logy=True)

def get_mu_std(samples):
    X = np.array(np.vstack([s.X for s in samples]), np.float64)
    mu = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mu, std

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--version', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=4)
    args = parser.parse_args()

    basedir = '/eos/uscms/store/group/lpcbacon/jkrupa/May20/'
    figsdir = 'plots/%s/'%(args.version)
    modeldir = 'models/evt/v%i/'%(args.version)

    samples = ['VectorDiJet115','QCD']
    samples = [Sample(s, basedir, len(samples)) for s in samples]
    n_inputs = samples[0].X.shape[1]
    n_hidden = 4

    print 'Standardizing...'
    mu, std = get_mu_std(samples)
    [s.standardize(mu, std) for s in samples]

    model = ClassModel(n_inputs, n_hidden, len(samples))
    if args.train:
        print 'Training...'
        model.train(samples)
        model.save_as_keras(modeldir+'/weights.h5')
        model.save_as_tf(modeldir+'/graph.pb')
    else:
        print 'Loading...'
        model.load_model(modeldir+'weights.h5')

    if args.plot:
        print 'Inferring...'
        for s in samples:
            s.infer(model)

        samples.reverse()
        if REGRESSION:
            plot(np.linspace(60, 160, 20),
                 lambda s : s.Yhat[s.vidx][:,0],
                 samples, figsdir+'mass_regressed', xlabel='Regressed mass')
            plot(np.linspace(60, 160, 20),
                 lambda s : s.Y[s.vidx],
                 samples, figsdir+'mass_truth', xlabel='True mass')
        else:
            for i in xrange(len(samples) if MULTICLASS else 2):
                plot(np.linspace(0, 1, 50), 
                     lambda s, i=i : s.Yhat[s.vidx,i],
                     samples, figsdir+'class_%i'%i, xlabel='Class %i'%i)
#
#        for i in xrange(n_inputs):
#            plot(np.linspace(-2, 2, 20),
#                 lambda s, i=i : s.X[s.vidx,i],
#                 samples, figsdir+'feature_%i'%i, xlabel='Feature %i'%i)
#
