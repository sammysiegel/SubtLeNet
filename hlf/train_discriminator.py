#!/usr/bin/env python

from sys import exit 
from os import environ
environ['KERAS_BACKEND'] = 'theano'

import numpy as np
from utils import create_roc, Tagger


from keras.layers import Input, Dense, Dropout, Activation
from keras.models import Model 
from keras.callbacks import ModelCheckpoint 
from keras.optimizers import Adam 
from keras.utils import np_utils

def reform(arr, train_frac, val_frac, fields, weight, label, bdt_name=None):
    n = arr.shape[0] 
    ns = {}
    ns['train'] = (0,int(train_frac*n))
    ns['val'] = (ns['train'][1],ns['train'][1]+int(val_frac*n))
    ns['test'] = (ns['val'][1],n)
    print 'label=%i, n_train=%i, n_val=%i, n_test=%i'%(label,ns['train'][1],ns['val'][1]-ns['val'][0],ns['test'][1]-ns['test'][0])
    weight_norm = 100. / np.sum(arr[weight])
    x = {}; y = {}; w = {}; bdt = {}
    for subset in ['train','val','test']:
        n_ = ns[subset]
        x[subset] = arr[fields].view(np.float64).reshape(arr[fields].shape+(-1,))[n_[0]:n_[1]]
        w[subset] = arr[weight][n_[0]:n_[1]] * weight_norm 
        y[subset] = (label * np.ones(n_[1]-n_[0])).astype(int)
        if bdt_name:
            bdt[subset] = arr[bdt_name][n_[0]:n_[1]]
    return {'x':x,'y':y,'w':w,'bdt':bdt}
    

def load_data(train_frac,val_frac,fields):
    # arr_bkg = np.load('../data/Background_selected.npy')
    # arr_sig = np.load('../data/Top_selected.npy')
    arr_bkg = np.load('../data/QCD_goodjets.npy')
    arr_sig = np.load('../data/ZpTT_goodjets.npy')

    np.random.shuffle(arr_bkg)
    np.random.shuffle(arr_sig)

    bkg = reform(arr_bkg,train_frac,val_frac,fields,'weight',0,'top_ecf_bdt')
    sig = reform(arr_sig,train_frac,val_frac,fields,'weight',1,'top_ecf_bdt')

    x = {}; y = {}; w = {}; bdt = {}

    for subset in ['train','val','test']:
        x[subset] = np.concatenate((bkg['x'][subset],sig['x'][subset]), axis=0)
        w[subset] = np.concatenate((bkg['w'][subset],sig['w'][subset]), axis=0)
        bdt[subset] = np.concatenate((bkg['bdt'][subset],sig['bdt'][subset]), axis=0)
        y_vec = np.concatenate((bkg['y'][subset],sig['y'][subset]), axis=0)
        y[subset] = np_utils.to_categorical(y_vec, 2)

    return x,y,w,bdt

fields = ['tau21sd','tau32sd'] + ['ecf%i'%i for i in xrange(11)]
x,y,w,bdt = load_data(0.5,0.25,fields)


dim = x['train'].shape[1]


inputs = Input(shape=(dim,), name='hlf')

l = Dense(32, activation='relu')(inputs)
l = Dense(32, activation='relu')(l)
l = Dense(16, activation='relu')(l)
output = Dense(2, activation='softmax', name='hlf_disc')(l)

model = Model(inputs=inputs, outputs=output)
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print model.summary()

checkpoint = ModelCheckpoint(filepath='simple_disc.h5', save_best_only=True)
model.fit(x['train'], y['train'], sample_weight=w['train'],
          batch_size=32, epochs=1, verbose=1,
          validation_data=(x['val'],y['val'],w['val']), 
          shuffle=True,
          callbacks=[checkpoint])


y_pred = model.predict(x['test'])
test_accuracy = np.sum(
                    (np.argmax(y['test'], axis=1)==np.argmax(y_pred, axis=1))
                )/float(x['test'].shape[0])

print 'DNN accuracy = %.3g'%(test_accuracy)

score = model.evaluate(x['test'], y['test'], batch_size=32, verbose=1, sample_weight=w['test'])

print '' 
print 'DNN score =',score

dnn = Tagger(y_pred[:,1], 'DNN', 0, 1, False)
bdt = Tagger(bdt['test'], 'BDT', -1, 1, False)
create_roc([dnn,bdt],np.argmax(y['test'],axis=1),w['test'],'simple')
