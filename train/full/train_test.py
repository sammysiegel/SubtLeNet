#!/usr/local/bin/python2.7

import sys 

from sys import exit 
from os import environ, system

environ['KERAS_BACKEND'] = 'tensorflow'

import signal
import utils
import numpy as np

from keras.layers import Input, Dense
from keras.models import Model 
from keras.callbacks import ModelCheckpoint, LambdaCallback, TensorBoard
from keras.optimizers import Adam, SGD
from keras.utils import np_utils

import adversarial
import obj 
import config
config.DEBUG = False

from keras import backend as K
K.set_image_data_format('channels_last')


def make_coll(fpath):
    coll = obj.PFSVCollection()
    coll.add_categories(['singletons'], fpath) 
    return coll 

top_4 = make_coll('/data/t3serv014/snarayan/deep/v_deep_2//PARTITION/Top_*_CATEGORY.npy') # T
qcd_0 = make_coll('/data/t3serv014/snarayan/deep/v_deep_2//PARTITION/QCD_*_CATEGORY.npy') # T

data = [top_4, qcd_0]


# build the discriminator
inputs  = Input(shape=(len(obj.default_variables),), name='input')

dense = Dense(10, activation='tanh', kernel_initializer='lecun_uniform')(inputs)
dense = Dense(10, activation='tanh', kernel_initializer='lecun_uniform')(dense)

category_pred =  Dense(config.n_truth, activation='softmax', kernel_initializer='lecun_uniform')(dense)

classifier = Model(inputs=inputs, outputs=category_pred)
classifier.compile(optimizer=Adam(lr=0.001),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])


classifier.summary()

classifier_train_gen = obj.generateSingletons(data, partition='train', batch=1000)


def save_model(name='classifier'):
    classifier.save('%s.h5'%name)

def save_and_exit(signal=None, frame=None):
    save_model()
    flog.close()
    exit(1)

# ctrl+C now triggers a graceful exit
signal.signal(signal.SIGINT, save_and_exit)


classifier.fit_generator(classifier_train_gen, 
                            steps_per_epoch=1000, 
                            epochs=2)

save_model('pretrained')
