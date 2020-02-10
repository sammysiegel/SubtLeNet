#!/usr/bin/env python
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.utils import shuffle
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
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

basedir = '/home/jeffkrupa/files/deepJet-v3'
Nqcd = 1000000
Nsig = 1000000
Nparts = 40

def _make_parent(path):
    os.system('mkdir -p %s'%('/'.join(path.split('/')[:-1])))

class Sample(object):
    def __init__(self, name, base, max_Y):
        self.name = name 
        self.Yhat = {} 

        N = Nqcd if 'QCD' in name else Nsig

        self.X = np.load('%s/%s_%s.npy'%(base, name, 'x'))[:N]
        self.SS = np.load('%s/%s_%s.npy'%(base, name, 'ss'))[:N]

        self.Y = np_utils.to_categorical((np.load('%s/%s_%s.npy'%(base, name, 'y'))[:N] > 0).astype(np.int), 2)
        self.W = np.load('%s/%s_%s.npy'%(base, name, args.wname))[:N]

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
        if 'GRU' in model.name: self.X = np.reshape(self.X, (self.X.shape[0], self.X.shape[1]/Nparts, Nparts)) 
        if 'Dense' in model.name: self.X = np.reshape(self.X, (self.X.shape[0],self.X.shape[1]))
        self.Yhat[model.name] = model.predict(self.X)
    def standardize(self, mu, std):
        self.X = (self.X - mu) / std

class ClassModel(object):
    def __init__(self, n_inputs, h_hidden, n_targets, samples, model, modeldir="."):
        self._hidden = 0
        self.name = model
        self.n_inputs = n_inputs
        self.n_targets = n_targets if MULTICLASS else 2
        self.n_hidden = n_hidden

        self.tX = np.vstack([s.X[:][s.tidx] for s in samples])
        self.tW = np.concatenate([s.W[s.tidx] for s in samples])
        #self.tD = np.concatenate([s.D[s.tidx] for s in samples])
        self.vX = np.vstack([s.X[:][s.vidx] for s in samples])
        self.vW = np.concatenate([s.W[s.vidx] for s in samples])
        #self.vD = np.concatenate([s.D[s.vidx] for s in samples])
      

        self.tY = np.vstack([s.Y[s.tidx] for s in samples])
        self.vY = np.vstack([s.Y[s.vidx] for s in samples])
        self.tSS = np.vstack([s.SS[s.tidx] for s in samples])
        self.vSS = np.vstack([s.SS[s.vidx] for s in samples])

        #for i in xrange(self.tY.shape[1]):
        #  tot = np.sum(self.tW[self.tY[:,i] == 1])
        #  self.tW[self.tY[:,i] == 1] *= 100/tot
        #  self.vW[self.vY[:,i] == 1] *= 100/tot


        if 'GRU' in self.name:
            self.tX = np.reshape(self.tX, (self.tX.shape[0], self.tX.shape[1]/Nparts, Nparts))
            self.vX = np.reshape(self.vX, (self.vX.shape[0], self.vX.shape[1]/Nparts, Nparts))

            print self.tX[0]
            self.inputs = Input(shape=(self.tX.shape[1], Nparts), name='input')
            h = self.inputs
        
            NPARTS=20
            CLR=0.001
            LWR=0.1
    
            h = BatchNormalization(momentum=0.6)(h)
            gru = GRU(140,activation='relu',recurrent_activation='hard_sigmoid',name='gru_base',dropout=0.05)(h)
            dense   = Dense(200, activation='relu')(gru)
            norm    = BatchNormalization(momentum=0.6, name='dense4_bnorm')  (dense)
            dense   = Dense(100, activation='relu')(norm)
            norm    = BatchNormalization(momentum=0.6, name='dense5_bnorm')  (dense)
            dense   = Dense(50, activation='relu')(norm)
            norm    = BatchNormalization(momentum=0.6, name='dense6_bnorm')  (dense)
            dense   = Dense(20, activation='relu')(dense)
            dense   = Dense(10, activation='relu')(dense)
            outputs = Dense(self.n_targets, activation='sigmoid')(norm)
            self.model = Model(inputs=self.inputs, outputs=outputs)

            self.model.compile(loss='binary_crossentropy', optimizer=Adam(CLR), metrics=['accuracy'])
            self.es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=3)

        if 'Dense' in self.name:
            self.inputs = Input(shape=(int(n_inputs),), name='input')
            h = self.inputs
            h = BatchNormalization(momentum=0.6)(h)
            for _ in xrange(n_hidden-1):
              h = Dense(int(n_inputs*0.2), activation='relu')(h)
              h = BatchNormalization()(h)
            h = Dense(int(n_inputs*0.2), activation='tanh')(h)
            h = BatchNormalization()(h)
            self.outputs = Dense(self.n_targets, activation='softmax', name='output')(h)
            self.model = Model(inputs=self.inputs, outputs=self.outputs)
            self.model.compile(optimizer=Adam(),
                               loss='binary_crossentropy', metrics=['accuracy'])
            self.es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5)
        self.model.summary()


    def train(self, samples,modeldir="."):
       
        print self.tX.shape, self.tY.shape, self.tW.shape 
        print self.vX.shape, self.vY.shape, self.vW.shape 
        history = self.model.fit(self.tX, self.tY, sample_weight=self.tW, 
                                 batch_size=1000, epochs=30, shuffle=True,
                                 validation_data=(self.vX, self.vY, self.vW),callbacks=[self.es])
        plt.clf()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(modeldir+'loss.png')
        plt.savefig(modeldir+'loss.pdf')

        with open('history.log','w') as flog:
            history = history.history
            flog.write(','.join(history.keys())+'\n')
            for l in zip(*history.values()):
                flog.write(','.join([str(x) for x in l])+'\n')

    def weights(self):
        return self.tW,self.vW
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


def get_mu_std(samples, modeldir):
    X = np.array(np.vstack([s.X for s in samples]), np.float64)
    mu = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    np.save('standardize_mu.npy',mu)
    np.save('standardize_std.npy',std)

    for it,val in enumerate(np.nditer(mu)):
        if val == 0.: mu[it] = 1.
    for it,val in enumerate(np.nditer(std)):
        if val == 0.: std[it] = 1.

    np.save(modeldir+'standardize_mu.npy',mu)
    np.save(modeldir+'standardize_std.npy',std)
    
    return mu, std


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--version', type=int, default=0)
    parser.add_argument('--wname', type=str, default="w")

    args = parser.parse_args()
    print "using weight %s"%args.wname 

    figsdir = 'plots/%s/'%(args.version)
    modeldir = 'models/evt/v%i/'%(args.version)

    _make_parent(modeldir)
    SIG = 'VectorDiJet-flat'
    BKG = 'QCD'

    models = ['GRU',]

    samples = [SIG,BKG]

    samples = [Sample(s, basedir, len(samples)) for s in samples]
    n_inputs = samples[0].X.shape[1]
    print('# sig: ',samples[0].X.shape[0], '#bkg: ',samples[1].X.shape[0])

    print 'Standardizing...'
    #mu, std = get_mu_std(samples,modeldir)
    #[s.standardize(mu, std) for s in samples]

    n_hidden = 5
    if 'Dense' in models:
        modelDNN = ClassModel(n_inputs, n_hidden, len(samples),samples,'Dense',modeldir)
        if args.train:
            print 'Training dense...'
            modelDNN.train(samples,modeldir)
            modelDNN.save_as_keras(modeldir+'/weights_dense.h5')
            modelDNN.save_as_tf(modeldir+'/graph_dense.pb')
        else:
            print 'Loading dense...'
            modelDNN.load_model(modeldir+'weights_dense.h5')

        if args.plot:
            for s in samples:
              s.infer(modelDNN)
      
        del modelDNN

    if 'GRU' in models:
        modelGRU = ClassModel(n_inputs, n_hidden, len(samples),samples,'GRU')
        if args.train:
            print 'Training gru...'
            modelGRU.train(samples)
            modelGRU.save_as_keras(modeldir+'/weights_gru.h5')
            modelGRU.save_as_tf(modeldir+'/graph_gru.pb')
        else:
            print 'Loading gru...'
            modelGRU.load_model(modeldir+'weights_gru.h5')
        if args.plot:
            for s in samples:
              s.infer(modelGRU)
        del modelGRU

    if args.plot:

        samples.reverse()
        roccer_hists = {}
        roccer_hists_SS = {}
        SS_vars = {'N2':1,'deepTagZqq':2}

        sig_hists = {}
        bkg_hists = {}

        for idx,num in SS_vars.iteritems():
                     roccer_hists_SS[idx] = plot(np.linspace(0,1,50),
                     lambda s: s.SS[s.vidx,num-1],
                     samples, figsdir+'%s'%(idx), xlabel='%s'%(idx))
       
        sig_hists['N2'] = roccer_hists_SS['N2'][SIG]    
        bkg_hists['N2'] = roccer_hists_SS['N2'][BKG]    
        sig_hists['deepTagZqq'] = roccer_hists_SS['deepTagZqq'][SIG]  
        bkg_hists['deepTagZqq'] = roccer_hists_SS['deepTagZqq'][BKG]    

        for model in models:
            for i in xrange(len(samples) if MULTICLASS else 2):

                roccer_hists = plot(np.linspace(0, 1, 50), 
                       lambda s, i=i : s.Yhat[model][s.vidx,i], 
                       samples, figsdir+'class_%i_%s'%(i,model), xlabel='Class %i %s'%(i,model))


                sig_hists[model] = roccer_hists[SIG]
                bkg_hists[model] = roccer_hists[BKG] 

        r1 = utils.Roccer(y_range=range(0,1),axis=[0,1,0,1],)
        r1.clear()

        r1.add_vars(sig_hists,           
                    bkg_hists,
                    {'Dense':'Dense',
		     'GRU':'GRU',
                     'N2':'N2',
                     'tau21':'tau21'}
        )
        r1.plot(figsdir+'class_%s_ROC'%(str(args.version)))                 
