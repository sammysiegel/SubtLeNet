#!/usr/local/bin/python2.7

from sys import exit 
from os import environ
environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import utils


from keras.layers import Input, Dense, Dropout, Activation, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, LSTM, Convolution1D, MaxPooling1D, MaxPooling1D
from keras.models import Model 
from keras.callbacks import ModelCheckpoint 
from keras.optimizers import Adam 
from keras.utils import np_utils
import obj 
obj.DEBUG = True

from keras import backend as K
K.set_image_data_format('channels_last')


def make_coll(fpath):
    coll = obj.PFSVCollection()
    coll.add_classes(['singletons', 'charged', 'inclusive', 'sv'], fpath) 
    return coll 

top_4 = make_coll('/home/snarayan/hscratch/baconarrays/v6/RSGluonToTT_*_4_XXXX.npy') # T
# top_2 = make_coll('/home/snarayan/hscratch/baconarrays/v4/RSGluonToTT_*_2_XXXX.npy') # W
qcd_0 = make_coll('/home/snarayan/hscratch/baconarrays/v6/QCD_*_0_XXXX.npy') # q/g

data = [top_4, qcd_0]
# data = [top_4, top_2, qcd_0]
# data = [qcd_1, qcd_2, top_1, top_2, top_3]

# preload some data just to get the dimensions
data[0].objects['charged'].load(memory=False)
data[0].objects['inclusive'].load(memory=False)
data[0].objects['sv'].load(memory=False)

# charged layer
dims = data[0].objects['charged'].data.shape
input_charged = Input(shape=(dims[1], dims[2]), name='input_charged')
conv = Convolution1D(32, 1, padding='valid', activation='relu', input_shape=(dims[1],dims[2]))(input_charged)
conv = Convolution1D(16, 1, padding='valid', activation='relu')(conv)
conv = Convolution1D(8, 1, padding='valid', activation='relu')(conv)
conv = Convolution1D(4, 1, padding='valid', activation='relu')(conv)
lstm_charged = LSTM(20)(conv)

# inclusive layer
dims = data[0].objects['inclusive'].data.shape 
input_inclusive = Input(shape=(dims[1], dims[2]), name='input_inclusive')
conv = Convolution1D(32, 1, padding='valid', activation='relu', input_shape=(dims[1],dims[2]))(input_inclusive)
conv = Convolution1D(16, 1, padding='valid', activation='relu')(conv)
conv = Convolution1D(8, 1, padding='valid', activation='relu')(conv)
conv = Convolution1D(4, 1, padding='valid', activation='relu')(conv)
lstm_inclusive = LSTM(20)(conv)

# sv layer
dims = data[0].objects['sv'].data.shape 
input_sv = Input(shape=(dims[1], dims[2]), name='input_sv')
conv = Convolution1D(32, 1, padding='valid', activation='relu', input_shape=(dims[1],dims[2]))(input_sv)
conv = Convolution1D(16, 1, padding='valid', activation='relu')(conv)
conv = Convolution1D(8, 1, padding='valid', activation='relu')(conv)
conv = Convolution1D(4, 1, padding='valid', activation='relu')(conv)
lstm_sv = LSTM(20)(conv)


# merge
merge = concatenate([lstm_charged, lstm_inclusive, lstm_sv])
output_p = Dense(5, activation='softmax')(merge)
output_b = Dense(10, activation='softmax')(merge)


# model = Model(inputs=[input_charged, input_inclusive, input_sv], outputs=[output_p, output_b])
model = Model(inputs=[input_charged, input_inclusive, input_sv], outputs=[output_p])
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print model.summary()

train_generator = obj.generatePFSV(data, partition='train', batch=100)
validation_generator = obj.generatePFSV(data, partition='validate', batch=10000)

# x, y, w = next(train_generator)
# model.fit(x, y, sample_weight=w, epochs=1, batch_size=32, verbose=1)

try:
  model.fit_generator(train_generator, 
                      steps_per_epoch=10000, 
                      epochs=10, 
                      validation_data=validation_generator, 
                      validation_steps=100)
except StopIteration:
  pass

model.save('model2.h5')


# model.fit(x['train'], y['train'], sample_weight=w['train'],
#           batch_size=30, epochs=1, verbose=1,
#           validation_data=(x['val'],y['val'],w['val']), 
#           shuffle=True)


# y_pred = model.predict(x['test'])
# test_accuracy = np.sum(
#                     (np.argmax(y['test'], axis=1)==np.argmax(y_pred, axis=1))
#                 )/float(x['test'].shape[0])

# print 'NN accuracy = %.3g'%(test_accuracy)

# score = model.evaluate(x['test'], y['test'], batch_size=32, verbose=1, sample_weight=w['test'])

# print '' 
# print 'NN score =',score
