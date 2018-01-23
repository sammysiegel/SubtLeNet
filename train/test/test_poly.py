#!/usr/local/bin/python2.7

from sys import exit 
from os import environ, system
# environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# environ["CUDA_VISIBLE_DEVICES"] = ""

environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import signal
from adversarial import PolyLayer

from keras.layers import Input, Dense
from keras.models import Model 
from keras.callbacks import ModelCheckpoint, LambdaCallback, TensorBoard
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras import backend as K
K.set_image_data_format('channels_last')


# build test data 
N = 100000
X = np.random.rand(N) 
y = np.reshape(4 + 2*np.power(X, 1) + 7*np.power(X, 2), (N,))



inputs = Input(shape=(1,), name='inputs')
poly = PolyLayer(2)(inputs)

model = Model(inputs=inputs, outputs=poly)
model.compile(optimizer=Adam(lr=0.0005),
              loss='mse')

model.summary()

for _ in xrange(10):
    model.fit(X, y, batch_size=32)

for layer in model.layers:
    weights = layer.get_weights() 
    print weights 