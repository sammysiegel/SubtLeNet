#!/usr/local/bin/python2.7

from sys import exit 
from os import environ
environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import utils


from keras.layers import Input, Dense, Dropout, Activation, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.models import Model 
from keras.callbacks import ModelCheckpoint 
from keras.optimizers import Adam 
from keras.utils import np_utils

from keras import backend as K
K.set_image_data_format('channels_last')


def load_data(train_frac,val_frac):
    arr_z = np.load('/home/snarayan/hscratch/redpanda/camera_v2/ZpTT_med-1250_4_0_gen.npy')
    arr_t = np.load('/home/snarayan/hscratch/redpanda/camera_v2/ZpTT_med-1250_4_0_truth.npy')

    x = {}; y = {}    

    N = arr_z.shape[0]

    arr_z = arr_z.reshape(arr_z.shape + (1,))
    arr_t = arr_t.reshape(arr_t.shape + (1,))

    Ntrain = int(N*train_frac)
    Nval = Ntrain + int(N*val_frac) 
    

    x['train'] = arr_z[:Ntrain]
    y['train'] = arr_t[:Ntrain]

    x['val'] = arr_z[Ntrain:Nval]
    y['val'] = arr_t[Ntrain:Nval]

    x['test'] = arr_z[Nval:]
    y['test'] = arr_t[Nval:]

    return x,y

x,y = load_data(0.5,0.25)


dim0 = x['train'].shape[1]
dim1 = x['train'].shape[2]


inputs = Input(shape=(dim0, dim1, 1), name='z')

conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv2), conv1], axis=3)
conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

conv10 = Conv2D(1, (1, 1), activation='softmax')(conv9)

model = Model(inputs=inputs, outputs=conv10)
model.compile(optimizer=Adam(),
              loss=utils.dice_coef_loss,
              metrics=[utils.dice_coef])

print model.summary()

model.fit(x['train'], y['train'],
          batch_size=100, epochs=1, verbose=1,
          validation_data=(x['val'],y['val']), 
          shuffle=True)


y_pred = model.predict(x['test'])
test_accuracy = np.sum(
                    (np.argmax(y['test'], axis=1)==np.argmax(y_pred, axis=1))
                )/float(x['test'].shape[0])

print 'DNN accuracy = %.3g'%(test_accuracy)

score = model.evaluate(x['test'], y['test'], batch_size=32, verbose=1)

print '' 
print 'DNN score =',score
