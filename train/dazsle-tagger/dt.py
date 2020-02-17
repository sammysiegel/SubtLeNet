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
from keras import regularizers

import subtlenet.utils as utils 
#utils.set_processor('gpu')
VALSPLIT = 0.3 #0.7
MULTICLASS = False
REGRESSION = False
np.random.seed(10)

basedir = '/home/jeffkrupa/files/deepJet-v6'
Nqcd = 1000000
Nsig = 1000000
Nparts = 20

def newShape(X):
    Xnew = np.zeros((X.shape[0], Nparts, X.shape[1]/Nparts))
    for i in range(0,X.shape[0]):
        Xnew[i] = np.reshape(X[i],(X.shape[1]/Nparts,Nparts)).T
    return Xnew

def _make_parent(path):
    os.system('mkdir -p %s'%('/'.join(path.split('/')[:-1])))

class Sample(object):
    def __init__(self, name, base, max_Y):
        self.name = name 
        self.Yhat = {} 

        N = Nqcd if 'QCD' in name else Nsig

        self.X = np.load('%s/%s_%s%s.npy'%(base, name, 'x', args.inputtag))[:][:,:] 
        print self.X[0], self.X.shape
        self.SS = np.load('%s/%s_%s%s.npy'%(base, name, 'ss', args.inputtag))[:]
        self.K = np.load('%s/%s_%s%s.npy'%(base, name, 'w', args.inputtag))[:]
        self.Y = np_utils.to_categorical((np.load('%s/%s_%s.npy'%(base, name, 'y'))[:N] > 0).astype(np.int), 2)
        self.W = np.ones(self.Y.shape[0])

        print 'before', self.X.shape
        np.random.shuffle(self.X)
        np.random.shuffle(self.SS)
        np.random.shuffle(self.K)
        np.random.shuffle(self.Y)
        np.random.shuffle(self.W)
        print 'after', self.X.shape

        self.X = self.X[:N]
        self.SS = self.SS[:N]
        self.K = self.K[:N]
        self.Y = self.Y[:N]
        self.W = self.W[:N]

        #if 'QCD' in name: self.W = np.load('%s/%s_%s.npy'%(base, name, args.wname))[:N]
        self.idx = np.random.permutation(self.Y.shape[0])
        print self.Y
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

class ClassModel(object):
    def __init__(self, n_inputs, h_hidden, n_targets, samples, model, modeldir="."):
        self._hidden = 0
        self.name = model
        self.n_inputs = n_inputs
        self.n_targets = n_targets if MULTICLASS else 2
        self.n_hidden = n_hidden

        self.tX = np.vstack([s.X[:][s.tidx] for s in samples])
        self.tW = np.concatenate([s.W[s.tidx] for s in samples])
        self.vX = np.vstack([s.X[:][s.vidx] for s in samples])
        self.vW = np.concatenate([s.W[s.vidx] for s in samples])

        self.tK = np.vstack([s.K[:][s.tidx] for s in samples]) 
        self.vK = np.vstack([s.K[:][s.vidx] for s in samples]) 

        self.tY = np.vstack([s.Y[s.tidx] for s in samples])
        self.vY = np.vstack([s.Y[s.vidx] for s in samples])
        self.tSS = np.vstack([s.SS[s.tidx] for s in samples])
        self.vSS = np.vstack([s.SS[s.vidx] for s in samples])


        print self.tX[0]
        self.tX = newShape(self.tX)
        self.vX = newShape(self.vX)
        #self.tX = np.reshape(self.tX, (self.tX.shape[0], self.tX.shape[1]/Nparts, Nparts))
        #self.vX = np.reshape(self.vX, (self.vX.shape[0], self.vX.shape[1]/Nparts, Nparts))
        
        print 'after', self.tX[0]
        
        print self.tY[:100]
        print self.tW[:100]
        if 'GRU' in self.name:

            self.inputs = Input(shape=(self.tX.shape[1],self.tX.shape[2]), name='input')
            #self.inputs = Input(shape=(self.tX.shape[1], Nparts), name='input')
            h = self.inputs
        
            NPARTS=20
            CLR=0.001
            LWR=0.1
            h = BatchNormalization(input_shape=(self.tX.shape[1],Nparts),momentum=0.6)(h)
            #gru = GRU(300,activation='relu',recurrent_activation='hard_sigmoid',name='gru_base',activity_regularizer=regularizers.l1(0.01))(h)
            gru = GRU(10,activation='relu',recurrent_activation='hard_sigmoid',name='gru_base',dropout=0.1)(h)
            #dense   = Dense(200, activation='relu')(gru)
            norm    = BatchNormalization(momentum=0.6, name='dense4_bnorm')  (gru)
            #dense   = Dense(100, activation='relu')(norm)
            #dropout = Dropout(0.2)(dense)# Dense(100, activation='relu')(norm)
            #norm    = BatchNormalization(momentum=0.6, name='dense5_bnorm')  (dropout)
            dense   = Dense(50, activation='relu')(norm)
            dropout = Dropout(0.2)(dense)# Dense(100, activation='relu')(norm)
            norm    = BatchNormalization(momentum=0.6, name='dense6_bnorm')  (dropout)
            dense   = Dense(20, activation='relu')(norm)
            dense   = Dense(10, activation='relu')(dense)
            outputs = Dense(self.n_targets, activation='sigmoid')(dense)
            self.model = Model(inputs=self.inputs, outputs=outputs)

            self.model.compile(loss='binary_crossentropy', optimizer=Adam(CLR), metrics=['accuracy'])
            self.es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=2)
            self.cp = ModelCheckpoint(modeldir+'/tmp.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')

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
            self.es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=2)

        self.model.summary()


    def plot(self,figdir):
        plt.clf()


        print self.tK[(self.tY[:,0]==1) & (self.tK[:,0] > 750.)][:100,0]
        print self.tW[(self.tY[:,0]==1) & (self.tK[:,0] > 750.)][:100]
        fig, ax = plt.subplots()
        plt.hist(self.tK[self.tY[:,0]==0][:,1],bins=40,range=(40,400),label='sig',alpha=0.5,normed=True)
        plt.hist(self.tK[self.tY[:,0]==1][:,1],bins=40,range=(40,400),label='bkg_weighted',alpha=0.5,weights=self.tW[self.tY[:,0]==1],normed=True)
        plt.hist(self.tK[self.tY[:,0]==1][:,1],bins=40,range=(40,400),label='bkg',alpha=0.5,normed=True)
        ax.set_xlabel('Jet mass (GeV)')
        plt.legend(loc='upper right')
        plt.savefig(figdir+'m.pdf')
        plt.savefig(figdir+'m.png')
        
        plt.clf()

        fig, ax = plt.subplots()
        plt.hist(self.tK[self.tY[:,0]==0][:,0],bins=40,range=(400,1500),label='sig',alpha=0.5,normed=True)
        plt.hist(self.tK[self.tY[:,0]==1][:,0],bins=40,range=(400,1500),label='bkg_weighted',alpha=0.5,weights=self.tW[self.tY[:,0]==1],normed=True)
        plt.hist(self.tK[self.tY[:,0]==1][:,0],bins=40,range=(400,1500),label='bkg',alpha=0.5,normed=True)
        ax.set_xlabel('Jet pt (GeV)')
        plt.legend(loc='upper right')
        plt.savefig(figdir+'pt.pdf')
        plt.savefig(figdir+'pt.png')

    def train(self, samples,modeldir="."):
       
        history = self.model.fit(self.tX, self.tY, sample_weight=self.tW, 
                                 batch_size=1000, epochs=100, shuffle=True,
                                 validation_data=(self.vX, self.vY, self.vW),callbacks=[self.es,self.cp])
        plt.clf()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(modeldir+'loss.png')
        plt.savefig(modeldir+'loss.pdf')
        plt.show()
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

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--version', type=int, default=0)
    parser.add_argument('--wname', type=str, default="w")
    parser.add_argument('--inputtag', type=str, default="")
    global args
    args = parser.parse_args()
    print "using weight %s"%args.wname 

    figsdir = 'plots/%s/'%(args.version)
    modeldir = 'models/evt/v%i/'%(args.version)

    _make_parent(figsdir)
    _make_parent(modeldir)
    SIG = 'VectorDiJet-flat'
    BKG = 'QCD'

    models = ['GRU',]

    samples = [SIG,BKG]

    samples = [Sample(s, basedir, len(samples)) for s in samples]
    n_inputs = samples[0].X.shape[1]
    print('# sig: ',samples[0].X.shape[0], '# bkg: ',samples[1].X.shape[0])


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
        modelGRU = ClassModel(n_inputs, n_hidden, len(samples),samples,'GRU',modeldir)
        modelGRU.plot(figsdir)
        if args.train:
            print 'Training gru...'
            modelGRU.train(samples,figsdir)
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
        #r1.clear()

        r1.add_vars(sig_hists,           
                    bkg_hists,
                    {'Dense':'Dense',
		     'GRU':'GRU',
                     'N2':'N2',
                     'deepZqq':'deepZqq'}
        )
        r1.plot(figsdir+'class_%s_ROC'%(str(args.version)))                 
