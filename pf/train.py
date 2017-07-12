#!/usr/local/bin/python2.7

from sys import exit 
from os import environ
#environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#environ["CUDA_VISIBLE_DEVICES"] = ""
environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import utils


from keras.layers import Input, Dense, Dropout, Activation, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, LSTM, Convolution1D, MaxPooling1D, MaxPooling1D
from keras.models import Model 
from keras.callbacks import ModelCheckpoint 
from keras.optimizers import Adam 
from keras.utils import np_utils

from keras import backend as K
K.set_image_data_format('channels_last')


def reform(arr, train_frac, val_frac, label):
    n = arr.shape[0] 
    ns = {}
    ns['train'] = (0,int(train_frac*n))
    ns['val'] = (ns['train'][1],ns['train'][1]+int(val_frac*n))
    ns['test'] = (ns['val'][1],n)
    weight_norm = 100. / n
    x = {}; y = {}; w = {}; extras = {}
    for subset in ['train','val','test']:
        n_ = ns[subset]
        x[subset] = arr[n_[0]:n_[1]]
        y[subset] = (label * np.ones(n_[1]-n_[0])).astype(int)
        w[subset] = weight_norm * np.ones(n_[1]-n_[0])
    return {'x':x,'y':y,'w':w}
    

def load_data(train_frac, val_frac):
    arr_sig = np.load('/mnt/hadoop/scratch/snarayan/redpanda/pf_v0/ZpTT_med-1500_38_0.npy')
    arr_bkg = np.load('/mnt/hadoop/scratch/snarayan/redpanda/pf_v0/QCD_ht1000to1500_3_0.npy')

    np.random.shuffle(arr_bkg)
    np.random.shuffle(arr_sig)

    bkg = reform(arr_bkg,train_frac,val_frac,0)
    sig = reform(arr_sig,train_frac,val_frac,1)

    x = {}; y = {}; w = {}

    for subset in ['train','val','test']:
        x[subset] = np.concatenate((bkg['x'][subset],sig['x'][subset]), axis=0)
        w[subset] = np.concatenate((bkg['w'][subset],sig['w'][subset]), axis=0)
        y_vec = np.concatenate((bkg['y'][subset],sig['y'][subset]), axis=0)
        y[subset] = np_utils.to_categorical(y_vec, 2)

    return x,y,w


x,y,w = load_data(0.5, 0.25)


dim0 = x['train'].shape[1]
dim1 = x['train'].shape[2]


inputs = Input(shape=(dim0, dim1), name='input')

conv = Convolution1D(32, 1, padding='valid', activation='relu', input_shape=(dim0,dim1))(inputs)
conv = Convolution1D(16, 1, padding='valid', activation='relu')(conv)
conv = Convolution1D(8, 1, padding='valid', activation='relu')(conv)
conv = Convolution1D(4, 1, padding='valid', activation='relu')(conv)

lstm = LSTM(100)(conv)

output = Dense(2, activation='softmax')(lstm)

model = Model(inputs=inputs, outputs=output)
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print model.summary()

model.fit(x['train'], y['train'], sample_weight=w['train'],
          batch_size=30, epochs=1, verbose=1,
          validation_data=(x['val'],y['val'],w['val']), 
          shuffle=True)


y_pred = model.predict(x['test'])
test_accuracy = np.sum(
                    (np.argmax(y['test'], axis=1)==np.argmax(y_pred, axis=1))
                )/float(x['test'].shape[0])

print 'NN accuracy = %.3g'%(test_accuracy)

score = model.evaluate(x['test'], y['test'], batch_size=32, verbose=1, sample_weight=w['test'])

print '' 
print 'NN score =',score
