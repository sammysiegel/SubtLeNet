#!/usr/bin/env python
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.utils import shuffle
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
#from subtlenet.backend.keras_objects import *
#from subtlenet.backend.losses import *
from keras.layers import Dense, BatchNormalization, Input, Dropout, Activation, concatenate, GRU
from keras.utils import np_utils
from keras.optimizers import Adam, Nadam, SGD
import keras.backend as K
from tensorflow.python.framework import graph_util, graph_io
import os, sys
import numpy as np
import pandas as pd
from collections import namedtuple

import subtlenet.utils as utils 
utils.set_processor('cpu')
VALSPLIT = 0.2 #0.7
MULTICLASS = False
REGRESSION = False
np.random.seed(5)

basedir = '/uscms/home/rbisnath/nobackup/training_plots/2019-05-28-lpchats-numpy-uproot-awkward/pkl_files'
Nqcd = 80000
def _make_parent(path):
    os.system('mkdir -p %s'%('/'.join(path.split('/')[:-1])))

class Sample(object):
    def __init__(self, name, base, max_Y):
        self.name = name 
 
        if 'Background' in name:        self.X = pd.read_pickle('%s/%s_%s.pkl'%(base, name, 'x')).values[:Nqcd]
        else:                    self.X = pd.read_pickle('%s/%s_%s.pkl'%(base, name, 'x')).values
        if 'Background' in name:        self.N2 = pd.read_pickle('%s/%s_%s.pkl'%(base, name, 'ss_vars')).values[:Nqcd]
        else:                    self.N2 = pd.read_pickle('%s/%s_%s.pkl'%(base, name, 'ss_vars')).values


        if REGRESSION:
            self.Y = pd.read_pickle('%s/%s_%s.pkl'%(base, name, 'y')).values[:,:1]
        else:
            if MULTICLASS:
                self.Y = np_utils.to_categorical(
                            pd.read_pickle('%s/%s_%s.pkl'%(base, name, 'y')).values[:,:1],
                            max_Y
                        )
            else:
              if 'Background' in name:
                self.Y = np_utils.to_categorical(
                            (pd.read_pickle('%s/%s_%s.pkl'%(base, name, 'y')).values[:Nqcd,:1] > 0).astype(np.int),
                            2
                        )
              else:
                self.Y = np_utils.to_categorical(
                            (pd.read_pickle('%s/%s_%s.pkl'%(base, name, 'y')).values[:,:1] > 0).astype(np.int),
                            2
                        )
 
        if 'Background' in name: self.W = pd.read_pickle('%s/%s_%s.pkl'%(base, name, 'w')).values.flatten()[:Nqcd]
        else:             self.W = pd.read_pickle('%s/%s_%s.pkl'%(base, name, 'w')).values.flatten()
        self.idx = np.random.permutation(self.X.shape[0]) #(self.Y.shape[0]) 
        #print 'self.name + idx: ', self.name, self.idx
        #print 'self.X.shape, self.Y.shape:', self.X.shape, self.Y.shape
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
        if 'GRU' in model.name: self.X = np.reshape(self.X, (self.X.shape[0], 1, self.X.shape[1])) 
        self.Yhat = model.predict(self.X)
    def standardize(self, mu, std):
        self.X = (self.X - mu) / std


class ClassModelDense(object):
    def __init__(self, n_inputs, n_hidden, n_targets,samples):
        self._hidden = 0
        self.name = 'Dense'
        self.n_inputs = n_inputs
        self.n_targets = n_targets if MULTICLASS else 2
        self.n_hidden = n_hidden
        self.inputs = Input(shape=(int(n_inputs),), name='input')
        h = self.inputs
        h = BatchNormalization(momentum=0.6)(h)
        for _ in xrange(n_hidden-1):
            h = Dense(int(n_inputs), activation='relu')(h)
            h = BatchNormalization()(h)
        h = Dense(int(n_inputs*0.1), activation='tanh')(h)
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
        self.tX = np.vstack([s.X[:][s.tidx] for s in samples])
        self.tW = np.concatenate([s.W[s.tidx] for s in samples])
        self.vX = np.vstack([s.X[:][s.vidx] for s in samples])
        self.vW = np.concatenate([s.W[s.vidx] for s in samples])

        if REGRESSION:
            self.tY = np.concatenate([s.Y[s.tidx] for s in samples])
            self.vY = np.concatenate([s.Y[s.vidx] for s in samples])
        else:
            self.tY = np.vstack([s.Y[s.tidx] for s in samples])
            self.vY = np.vstack([s.Y[s.vidx] for s in samples])
            self.tN2 = np.vstack([s.N2[s.tidx] for s in samples])
            self.vN2 = np.vstack([s.N2[s.vidx] for s in samples])
        #if not REGRESSION:
            #print '\n self.tY:', self.tY[:,0] != 0
            #print '\n self.tW:', self.tW
            #for i in xrange(self.tY.shape[1]):
                #tot = np.sum(self.tW[self.tY[:,i] != 0])
                #print 'tot:', tot
                #self.tW[self.tY[:,i] != 0] *= 100/tot
                #self.vW[self.vY[:,i] != 0] *= 100/tot

    def trainDense(self, samples):
        #print "tX.shp, tY.shp: ", self.tX.shape, self.tY.shape
        #print "vX.shp, vY.shp, vW.shp:", self.vX.shape, self.vY.shape, self.vW.shape
        print '\ntW: ', self.tW, '\n', np.isnan(self.tW)
        from functools import reduce
        print '\n', reduce((lambda a,b : a or b), np.isnan(self.tW))

        history = self.model.fit(self.tX, self.tY, sample_weight=self.tW, 
                                 batch_size=1000, epochs=40, shuffle=True,
                                 validation_data=(self.vX, self.vY, self.vW))
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

class ClassModelGRU(object):
    def __init__(self, n_inputs, n_hidden, n_targets,samples):
        self.tX = np.vstack([s.X[:][s.tidx] for s in samples])
        self.tW = np.concatenate([s.W[s.tidx] for s in samples])
        self.vX = np.vstack([s.X[:][s.vidx] for s in samples])
        self.vW = np.concatenate([s.W[s.vidx] for s in samples])
        self.name = 'GRU'
        print self.tX.shape

        self.tX = np.reshape(self.tX, (self.tX.shape[0], 1, self.tX.shape[1]))
        self.vX = np.reshape(self.vX, (self.vX.shape[0], 1, self.vX.shape[1]))

        print self.tX.shape
        self._hidden = 0

        self.n_inputs = n_inputs
        self.n_targets = n_targets if MULTICLASS else 2
        self.n_hidden = n_hidden
        self.inputs = Input(shape=(1,self.tX.shape[2]), name='input')
        h = self.inputs
        
        NPARTS=20
        CLR=0.01
        LWR=0.1
     
        gru = GRU(n_inputs,activation='relu',recurrent_activation='hard_sigmoid',name='gru_base')(h)
        dense   = Dense(100, activation='relu')(gru)
        norm    = BatchNormalization(momentum=0.6, name='dense4_bnorm')  (dense)
        dense   = Dense(50, activation='relu')(norm)
        norm    = BatchNormalization(momentum=0.6, name='dense5_bnorm')  (dense)
        dense   = Dense(20, activation='relu')(norm)
        dense   = Dense(10, activation='relu')(dense)
        outputs = Dense(self.n_targets, activation='sigmoid')(norm)
 
        self.model = Model(inputs=self.inputs, outputs=outputs)

        self.model.compile(loss='binary_crossentropy', optimizer=Adam(CLR), metrics=['accuracy'])

        self.tY = np.vstack([s.Y[s.tidx] for s in samples])
        self.vY = np.vstack([s.Y[s.vidx] for s in samples])
        self.tN2 = np.vstack([s.N2[s.tidx] for s in samples])
        self.vN2 = np.vstack([s.N2[s.vidx] for s in samples])
        if not REGRESSION:
            for i in xrange(self.tY.shape[1]):
                tot = np.sum(self.tW[self.tY[:,i] == 1])
                self.tW[self.tY[:,i] == 1] *= 100/tot
                self.vW[self.vY[:,i] == 1] *= 100/tot

    def trainGRU(self, samples):

        history = self.model.fit(self.tX, self.tY, sample_weight=self.tW, 
                                 batch_size=1000, epochs=10, shuffle=True,
                                 validation_data=(self.vX, self.vY, self.vW))
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
    p.clear()
    return hists


def get_mu_std(samples):
    X = np.array(np.vstack([s.X for s in samples]), np.float64)
    mu = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mu, std

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--dense', action='store_true')
    parser.add_argument('--gru', action='store_true')
    parser.add_argument('--model', type=str, default='DNN')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--version', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=2)
    args = parser.parse_args()

    basedir = basedir#'/eos/uscms/store/group/lpcbacon/jkrupa/May22_115_2/'
    figsdir = 'plots/%s/'%(args.version)
    modeldir = 'models/evt/v%i/'%(args.version)

    samples = ['BGHToWW','Background']
    samples = [Sample(s, basedir, len(samples)) for s in samples]
    n_inputs = samples[0].X.shape[1]
    print('# sig: ',samples[0].X.shape[0], '#bkg: ',samples[1].X.shape[0])
    n_hidden = 3

    #for s in samples: print s.name, 'tidx.shape + vidx.shape: ', s.tidx.shape, s.vidx.shape

    #print 'Standardizing...'
    #mu, std = get_mu_std(samples)
    #[s.standardize(mu, std) for s in samples]


    if 'DNN' in args.model:
        model = ClassModelDense(n_inputs, n_hidden, len(samples),samples)
        if args.train:
            print 'Training dense...'
            model.trainDense(samples)
            model.save_as_keras(modeldir+'/weights_dense.h5')
            model.save_as_tf(modeldir+'/graph_dense.pb')
        else:
            print 'Loading dense...'
            model.load_model(modeldir+'weights_dense.h5')

    if 'GRU' in args.model:
        model = ClassModelGRU(n_inputs, n_hidden, len(samples),samples)
        if args.train:
            print 'Training gru...'
            model.trainGRU(samples)
            model.save_as_keras(modeldir+'/weights_gru.h5')
            model.save_as_tf(modeldir+'/graph_gru.pb')
        else:
            print 'Loading gru...'
            model.load_model(modeldir+'weights_gru.h5')

    if args.plot:
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
            roccer_hists = {}
            roccer_hists_n = {}
            roccer_vars_n = {'N2':1}

            for i in xrange(len(samples) if MULTICLASS else 2):
                roccer_hists = plot(np.linspace(0, 1, 50), 
                     lambda s, i=i : s.Yhat[s.vidx,i],
                     samples, figsdir+'class_%i_%s'%(i,args.model), xlabel='Class %i %s'%(i,args.model))
  

                for idx,num in roccer_vars_n.iteritems():
                     roccer_hists_n[idx] = plot(np.linspace(0,1,50),
                     lambda s: s.N2[s.vidx,0],
                     samples, figsdir+'class_%i_%s'%(i,idx), xlabel='Class %i %s'%(i,idx))


            r1 = utils.Roccer(y_range=range(0,1),axis=[0,1,0,1])
            r1.clear()
            print roccer_hists
            sig_hists = {args.model:roccer_hists['BGHToWW'],
                'N2':roccer_hists_n['N2']['BGHToWW']}

            bkg_hists = {args.model:roccer_hists['Background'],
                'N2':roccer_hists_n['N2']['Background']}

            r1.add_vars(sig_hists,           
                        bkg_hists,
                        {args.model:args.model,
                         'N2':'N2'}
            )
            r1.plot(figsdir+'class_%s_%sROC'%(str(args.version),args.model))
