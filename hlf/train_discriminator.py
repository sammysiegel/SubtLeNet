#!/usr/bin/env python

from os import environ
environ['KERAS_BACKEND'] = 'theano'

import numpy as np

from keras.layers import Input, Dense, Dropout, Activation
from keras.models import Model 
from keras.callbacks import ModelCheckpoint 
from keras.optimizers import Adam 
from keras.utils import np_utils

def reform(arr, train_frac, val_frac, fields, weight, label):
    n = arr.shape[0] 
    ns = {}
    ns['train'] = (0,int(train_frac*n))
    ns['val'] = (ns['train'][1],ns['train'][1]+int(val_frac*n))
    ns['test'] = (ns['val'][1],n)
    weight_norm = 10000. / np.sum(arr[weight])
    x = {}; y = {}; w = {}
    for subset in ['train','val','test']:
        n_ = ns[subset]
        x[subset] = arr[fields].view(np.float64).reshape(arr[fields].shape+(-1,))[n_[0]:n_[1]]
        w[subset] = arr[weight][n_[0]:n_[1]] / weight_norm 
        y[subset] = (label * np.ones(n_[1]-n_[0])).astype(int)
    return {'x':x,'y':y,'w':w}
    

def load_data(train_frac,val_frac):
    fields = ['tau21sd','tau32sd'] + ['ecf%i'%i for i in xrange(11)]
    arr_bkg = np.load('../data/QCD_goodjets.npy')
    arr_sig = np.load('../data/ZpTT_goodjets.npy')

    np.random.shuffle(arr_bkg)
    np.random.shuffle(arr_sig)

    bkg = reform(arr_bkg,train_frac,val_frac,fields,'weight',0)
    sig = reform(arr_sig,train_frac,val_frac,fields,'weight',1)

    x = {}; y = {}; w = {}

    for subset in ['train','val','test']:
        x[subset] = np.concatenate((bkg['x'][subset],sig['x'][subset]), axis=0)
        w[subset] = np.concatenate((bkg['w'][subset],sig['w'][subset]), axis=0)
        y_vec = np.concatenate((bkg['y'][subset],sig['y'][subset]), axis=0)
        y[subset] = np_utils.to_categorical(y_vec, 2)

    return x,y,w,arr_bkg,arr_sig 

x,y,w,arr_bkg,arr_sig = load_data(0.5,0.25)


dim = x['train'].shape[1]


inputs = Input(shape=(dim,))

l = Dense(32, activation='relu')(inputs)
l = Dense(32, activation='relu')(l)
l = Dense(16, activation='relu')(l)
output = Dense(2, activation='softmax')(l)

model = Model(inputs=inputs, outputs=output)
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print model.summary()

checkpoint = ModelCheckpoint(filepath='simple_disc.h5', save_best_only=True)
model.fit(x['train'], y['train'],
          batch_size=32, epochs=1, verbose=1,
          validation_data=(x['val'],y['val']), 
          shuffle=True,
          callbacks=[checkpoint])


y_pred = model.predict(x['test'])
test_accuracy = np.sum(
                    np.argmax(y['test'], axis=1)==np.argmax(y_pred, axis=1)
                )/float(x['test'].shape[0])

print 'DNN accuracy = %.2f'%(test_accuracy)
