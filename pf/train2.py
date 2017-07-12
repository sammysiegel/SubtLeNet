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

from keras import backend as K
K.set_image_data_format('channels_last')


def make_coll(fpath):
    coll = obj.PFSVCollection()
    coll.add_classes(['singletons', 'charged', 'inclusive', 'sv'], fpath) 
    return coll 

qcd_1 = make_coll('/home/snarayan/hscratch/baconarrays/v13.2/QCD_*_1_XXXX.npy')
qcd_2 = make_coll('/home/snarayan/hscratch/baconarrays/v13.2/QCD_*_2_XXXX.npy')
top_1 = make_coll('/home/snarayan/hscratch/baconarrays/v13.2/BulkGravTohhTohbbhbb_*_1_XXXX.npy')
top_2 = make_coll('/home/snarayan/hscratch/baconarrays/v13.2/BulkGravTohhTohbbhbb_*_2_XXXX.npy')
top_3 = make_coll('/home/snarayan/hscratch/baconarrays/v13.2/BulkGravTohhTohbbhbb_*_3_XXXX.npy')

data = [qcd_1, top_2]
# data = [qcd_1, qcd_2, top_1, top_2, top_3]

# preload some data just to get the dimensions
qcd_1.objects['charged'].load(memory=False)
qcd_1.objects['inclusive'].load(memory=False)
qcd_1.objects['sv'].load(memory=False)

# charged layer
dims = qcd_1.objects['charged'].data.shape
input_charged = Input(shape=(dims[1], dims[2]), name='input_charged')
conv = Convolution1D(32, 1, padding='valid', activation='relu', input_shape=(dims[1],dims[2]))(input_charged)
conv = Convolution1D(16, 1, padding='valid', activation='relu')(conv)
conv = Convolution1D(8, 1, padding='valid', activation='relu')(conv)
conv = Convolution1D(4, 1, padding='valid', activation='relu')(conv)
lstm_charged = LSTM(20)(conv)

# inclusive layer
dims = qcd_1.objects['inclusive'].data.shape 
input_inclusive = Input(shape=(dims[1], dims[2]), name='input_inclusive')
conv = Convolution1D(32, 1, padding='valid', activation='relu', input_shape=(dims[1],dims[2]))(input_inclusive)
conv = Convolution1D(16, 1, padding='valid', activation='relu')(conv)
conv = Convolution1D(8, 1, padding='valid', activation='relu')(conv)
conv = Convolution1D(4, 1, padding='valid', activation='relu')(conv)
lstm_inclusive = LSTM(20)(conv)

# sv layer
dims = qcd_1.objects['sv'].data.shape 
input_sv = Input(shape=(dims[1], dims[2]), name='input_sv')
conv = Convolution1D(32, 1, padding='valid', activation='relu', input_shape=(dims[1],dims[2]))(input_sv)
conv = Convolution1D(16, 1, padding='valid', activation='relu')(conv)
conv = Convolution1D(8, 1, padding='valid', activation='relu')(conv)
conv = Convolution1D(4, 1, padding='valid', activation='relu')(conv)
lstm_sv = LSTM(20)(conv)


# merge
merge = concatenate([lstm_charged, lstm_inclusive, lstm_sv])
output_p = Dense(5, activation='softmax')(merge)
output_b = Dense(5, activation='softmax')(merge)


model = Model(inputs=[input_charged, input_inclusive, input_sv], outputs=[output_p, output_b])
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print model.summary()

train_generator = obj.generatePFSV(data, partition='train', batch=32)
validation_generator = obj.generatePFSV(data, partition='validate', batch=10000)

# x, y, w = next(train_generator)
# model.fit(x, y, sample_weight=w, epochs=1, batch_size=32, verbose=1)

model.fit_generator(train_generator, 
                    steps_per_epoch=1000, 
                    epochs=1, 
                    validation_data=validation_generator, 
                    validation_steps=100)


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
