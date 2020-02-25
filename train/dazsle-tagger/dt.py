#!/usr/bin/env python
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.utils import shuffle
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
#from subtlenet.backend.keras_objects import *
#from subtlenet.backend.losses import *
from keras.layers import Dense, BatchNormalization, Input, Dropout, Activation, concatenate, GRU,LSTM
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

basedir = '/eos/uscms/store/user/jkrupa/training/SVs-v1'
Nqcd = 1700000
Nsig = 1700000
Nparts = 30
NSVs = 5
def newShape(X,name='parts'):

    Ndims = Nparts if 'parts' in name else NSVs
    Xnew = np.zeros((X.shape[0], Ndims, X.shape[1]/Ndims))
    for i in range(0,X.shape[0]):
        Xnew[i] = np.reshape(X[i],(X.shape[1]/Ndims,Ndims)).T
    return Xnew

def select(X,N):
    np.random.shuffle(X)
    return X[:N]

def _make_parent(path):
    os.system('mkdir -p %s'%('/'.join(path.split('/')[:-1])))

class Sample(object):
    def __init__(self, name, base, max_Y):
        self.name = name 
        self.Yhat = {} 

        N = Nqcd if 'QCD' in name else Nsig

        self.X = np.load('%s/%s_%s%s.npy'%(base, name, 'x', args.inputtag))[:][:,:] 
        self.SS = np.load('%s/%s_%s%s.npy'%(base, name, 'ss', args.inputtag))[:]
        self.K = np.load('%s/%s_%s%s.npy'%(base, name, 'w', args.inputtag))[:]
        #self.Y = np_utils.to_categorical((np.load('%s/%s_%s.npy'%(base, name, 'y'))[:] > 0).astype(np.int), 2)
        self.Y = np.load('%s/%s_%s.npy'%(base, name, 'y'))[:]

   
        if args.SV:
          self.SV_X = np.load('%s/%s_%s.npy'%(base, name, 'SV_x'))[:]
          self.SV_Y = np_utils.to_categorical((np.load('%s/%s_%s.npy'%(base, name, 'SV_y'))[:] > 0).astype(np.int),3)[:,2]
   
          self.Y = np.concatenate((self.Y, self.SV_Y),axis=1)

        self.W = np.ones(self.Y.shape[0])

        self.X = select(self.X,N)
        self.SS = select(self.SS,N)
        self.K = select(self.K,N)
        self.Y = select(self.Y,N)
        self.W = select(self.W,N)
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
        if 'GRU' in model.name: 
          self.X = newShape(self.X)#np.reshape(self.X, (self.X.shape[0], self.X.shape[1]/Nparts, Nparts)) 
          if args.SV: self.SV_X = newShape(self.SV_X, 'SV')
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

        
        self.tX = newShape(self.tX)
        self.vX = newShape(self.vX)

        if args.SV: 
          self.tSV_X = np.vstack([s.SV_X[:][s.tidx] for s in samples])
          self.vSV_X = np.vstack([s.SV_X[:][s.vidx] for s in samples])
          self.tSV_X = newShape(self.tSV_X,'SV')
          self.vSV_X = newShape(self.vSV_X,'SV')
        
        if 'GRU' in self.name:

            self.inputs   = Input(shape=(self.tX.shape[1],self.tX.shape[2]), name='input')
            self.inputsSV = Input(shape=(self.tSV.shape[1],self.tSV.shape[2]), name='input')
            h = self.inputs
        
            NPARTS=20
            CLR=0.0005
            LWR=0.1

            gru = GRU(150)(self.inputs)#,activation='relu',recurrent_activation='sigmoid',name='gru_base',dropout=0.1)(h)
            if args.SV:
               gruSV = GRU(100)(self.inputsSV)
               combined = concatenate([gru.output,gruSV.output])

            if not (args.SV): dense   = Dense(200, activation='relu')(gru)
            else:             dense   = Dense(250, activation='relu')(combined)
            dense   = Dense(100, activation='relu')(dense)
            dense   = Dense(50, activation='relu')(dense)
            dense   = Dense(20, activation='relu')(dense)
            dense   = Dense(10, activation='relu')(dense)
            outputs = Dense(self.n_targets, activation='sigmoid')(dense)

            if not (args.SV): self.model = Model(inputs=self.inputs, outputs=outputs)
            else:             self.model = Model(inputs=[self.inputs,self.inputsSV], outputs=outputs)

         

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


        fig, ax = plt.subplots()
        plt.hist(self.tK[self.tY[:,0]==0][:,1],bins=40,range=(40,400),label='sig',alpha=0.5)
        plt.hist(self.tK[self.tY[:,0]==1][:,1],bins=40,range=(40,400),label='bkg_weighted',alpha=0.5,weights=self.tW[self.tY[:,0]==1])
        plt.hist(self.tK[self.tY[:,0]==1][:,1],bins=40,range=(40,400),label='bkg',alpha=0.5,normed=True)
        ax.set_xlabel('Jet mass (GeV)')
        plt.legend(loc='upper right')
        plt.savefig(figdir+'m.pdf')
        plt.savefig(figdir+'m.png')
        
        plt.clf()

        fig, ax = plt.subplots()
        plt.hist(self.tK[self.tY[:,0]==0][:,0],bins=40,range=(400,1500),label='sig',alpha=0.5)
        plt.hist(self.tK[self.tY[:,0]==1][:,0],bins=40,range=(400,1500),label='bkg_weighted',alpha=0.5,weights=self.tW[self.tY[:,0]==1])
        plt.hist(self.tK[self.tY[:,0]==1][:,0],bins=40,range=(400,1500),label='bkg',alpha=0.5,normed=True)
        ax.set_xlabel('Jet pt (GeV)')
        plt.legend(loc='upper right')
        plt.savefig(figdir+'pt.pdf')
        plt.savefig(figdir+'pt.png')

    def train(self, samples,modeldir="."):
    
        if not args.SV:    
           history = self.model.fit(self.tX, self.tY, sample_weight=self.tW, 
                                 batch_size=1000, epochs=50, shuffle=True,
                                 validation_data=(self.vX, self.vY, self.vW),
                                 callbacks=[self.es,self.cp])
        else:    
           history = self.model.fit([self.tX, self.tSV],
                                 self.tY, sample_weight=self.tW, 
                                 batch_size=1000, epochs=50, shuffle=True,
                                 validation_data=([self.vX, self.vSV], self.vY, self.vW),
                                 callbacks=[self.es,self.cp])
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
    parser.add_argument('--tmp', action='store_true')
    parser.add_argument('--SV', action='store_true')
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

            if not args.tmp: modelGRU.load_model(modeldir+'weights_gru.h5')
            else: modelGRU.load_model(modeldir+'tmp.h5')
        if args.plot:
            for s in samples:
              s.infer(modelGRU)
        del modelGRU

    import shutil
    shutil.copyfile("/home/jeffkrupa/SubtLeNet/train/dazsle-tagger/dt.py","/home/jeffkrupa/SubtLeNet/train/dazsle-tagger/"+modeldir+"dt_v%i.py"%args.version)
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
