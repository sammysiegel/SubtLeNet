#!/usr/local/bin/python2.7

from sys import exit 
from os import environ, system
environ['KERAS_BACKEND'] = 'tensorflow'
environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import utils
import signal

from keras.layers import Input, Dense
from keras.models import Model 
from keras.optimizers import Adam
from keras.utils import np_utils
from keras import backend as K
K.set_image_data_format('channels_last')

import obj 
import config 
config.DEBUG = False
config.n_truth = 5
config.truth = 'resonanceType'

''' 
instantiate data loaders 
''' 

def make_coll(fpath):
    coll = obj.PFSVCollection()
    coll.add_categories(['singletons'], fpath) 
    return coll 

top = make_coll('/home/snarayan/scratch5/baconarrays/v12_repro/PARTITION/ZprimeToTTJet_4_*_CATEGORY.npy')
qcd = make_coll('/home/snarayan/scratch5/baconarrays/v12_repro/PARTITION/QCD_0_*_CATEGORY.npy') 

data = [top, qcd]

# preload some data just to get the dimensions
data[0].objects['train']['singletons'].load(memory=False)
dims = data[0].objects['train']['singletons'].data.shape 
dims = (None,3,)

'''
first build the classifier!
'''

# set up data 
variables = ['tau32', 'tau21', 'msd']
classifier_train_gen = obj.generateSingletons(data, variables, partition='train', batch=100)
classifier_validation_gen = obj.generateSingletons(data, variables, partition='validate', batch=100)
classifier_test_gen = obj.generateSingletons(data, variables, partition='test', batch=1000)
test_i, test_o, test_w = next(classifier_test_gen)

inputs  = Input(shape=(dims[1],), name='input')
dense   = Dense(32, activation='relu',name='dense1',kernel_initializer='lecun_uniform') (inputs)
dense   = Dense(16, activation='tanh',name='dense2',kernel_initializer='lecun_uniform') (dense)
dense   = Dense(9, activation='tanh',name='dense3',kernel_initializer='lecun_uniform')  (dense)
y_hat   = Dense(config.n_truth, activation='softmax')                                   (dense)

classifier = Model(inputs=inputs, outputs=y_hat)
classifier.compile(optimizer=Adam(lr=0.001),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

print '########### CLASSIFIER ############'
classifier.summary()
print '###################################'


# ctrl+C now triggers a graceful exit
def save_classifier(name='tauNmsd', model=classifier):
    model.save('%s.h5'%name)



classifier.fit_generator(classifier_train_gen, 
                         steps_per_epoch=5000, 
                         epochs=10,
                         validation_data=classifier_validation_gen,
                         validation_steps=1000)


print test_o[0][5:]
print test_i[0][5:]
print classifier.predict(test_i[0][5:])

print test_o[0][:-5]
print test_i[0][:-5]
print classifier.predict(test_i[0][:-5])


save_classifier()
