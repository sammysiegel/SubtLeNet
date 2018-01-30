#!/usr/local/bin/python2.7

from sys import exit, stdout, argv
from os import environ, system
environ['KERAS_BACKEND'] = 'tensorflow'
environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import signal

from keras.layers import Input, Dense, Dropout, concatenate, LSTM, BatchNormalization, Conv1D, concatenate
from keras.models import Model 
from keras.callbacks import ModelCheckpoint, LambdaCallback, TensorBoard
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras import backend as K
K.set_image_data_format('channels_last')

from subtlenet import config 
from subtlenet.generators.gen_singletons import make_coll, generate

''' 
some global definitions
''' 

NEPOCH = 20
APOSTLE = 'v0'
system('cp %s shallow_models/train_%s.py'%(argv[0], APOSTLE))

''' 
instantiate data loaders 
''' 
basedir = '/data/t3serv014/snarayan/deep/v_deepgen_1/'
top = make_coll(basedir + '/PARTITION/Top_*_CATEGORY.npy')
hig = make_coll(basedir + '/PARTITION/Higgs_*_CATEGORY.npy')
qcd = make_coll(basedir + '/PARTITION/QCD_*_CATEGORY.npy')

data = [top, hig, qcd]

'''
first build the classifier!
'''

# set up data 
classifier_train_gen = generate(data, partition='train', batch=1000)
classifier_validation_gen = generate(data, partition='validate', batch=10000)
classifier_test_gen = generate(data, partition='test', batch=10)
test_i, test_o, test_w = next(classifier_test_gen)
#print test_i

inputs  = Input(shape=(len(config.gen_default_variables),), name='input')
dense   = Dense(32, activation='tanh',name='dense1',kernel_initializer='lecun_uniform') (inputs)
dense   = Dense(16, activation='tanh',name='dense2',kernel_initializer='lecun_uniform') (dense)
dense   = Dense(9, activation='tanh',name='dense3',kernel_initializer='lecun_uniform')  (dense)
y_hat   = Dense(config.n_truth, activation='softmax')                                   (dense)

classifier = Model(inputs=inputs, outputs=[y_hat])
classifier.compile(optimizer=Adam(lr=0.0005),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

print '########### CLASSIFIER ############'
classifier.summary()
print '###################################'


# ctrl+C now triggers a graceful exit
def save_classifier(name='shallow', model=classifier):
    model.save('shallow_models/%s_%s.h5'%(name, APOSTLE))

def save_and_exit(signal=None, frame=None, name='shallow', model=classifier):
    save_classifier(name, model)
    flog.close()
    exit(1)

signal.signal(signal.SIGINT, save_and_exit)



classifier.fit_generator(classifier_train_gen, 
                         steps_per_epoch=5000, 
                         epochs=NEPOCH,
                         validation_data=classifier_validation_gen,
                         validation_steps=10,
                        )
save_classifier()

