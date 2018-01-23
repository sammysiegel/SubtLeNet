#!/usr/local/bin/python2.7

from sys import exit, stdout, argv
from os import environ, system
environ['KERAS_BACKEND'] = 'tensorflow'
environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import utils
import signal

from keras.layers import Input, Dense, Dropout, concatenate, LSTM, BatchNormalization, Conv1D, concatenate
from keras.models import Model 
from keras.callbacks import ModelCheckpoint, LambdaCallback, TensorBoard
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras import backend as K
K.set_image_data_format('channels_last')

from adversarial import Adversary
import obj 
import config 
#config.DEBUG = True

#config.n_truth = 5
#config.truth = 'resonanceType'
#config.adversary_mask = 0

''' 
some global definitions
''' 

DECORRMASS = True
DECORRRHO = False
DECORRPT = False

adv_loss_weights = [0.0001, 200]

ADV = 2
NEPOCH = 3
APOSTLE = 'panda_5_decorr'
system('cp %s models/train_shallow_%s.py'%(argv[0], APOSTLE))

''' 
instantiate data loaders 
''' 

def make_coll(fpath):
    coll = obj.PFSVCollection()
    coll.add_categories(['singletons'], fpath) 
    return coll 

top = make_coll('/fastscratch/snarayan/pandaarrays/v1//PARTITION/Top_*_CATEGORY.npy')
qcd = make_coll('/fastscratch/snarayan/pandaarrays/v1//PARTITION/QCD_*_CATEGORY.npy')

data = [top, qcd]

'''
first build the classifier!
'''

# set up data 
classifier_train_gen = obj.generateSingletons(data, partition='train', batch=1000)
classifier_validation_gen = obj.generateSingletons(data, partition='validate', batch=10000)
classifier_test_gen = obj.generateSingletons(data, partition='validate', batch=10)
opts = {
        'decorr_mass' : DECORRMASS,
        'decorr_pt' : DECORRPT,
        'decorr_rho' : DECORRRHO,
        }
adv_train_gen = obj.generateSingletons(data, partition='train', batch=1000,**opts)
adv_validation_gen = obj.generateSingletons(data, partition='validate', batch=10000,**opts)
adv_test_gen = obj.generateSingletons(data, partition='validate', batch=10,**opts)
test_i, test_o, test_w = next(classifier_test_gen)
#print test_i

inputs  = Input(shape=(len(obj.default_variables),), name='input')
dense   = Dense(32, activation='tanh',name='dense1',kernel_initializer='lecun_uniform') (inputs)
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

''' 
now build the adversarial setup
'''

# build the model 
kin_hats = Adversary(config.n_decorr_bins, 
                     n_outputs=(int(DECORRMASS)+int(DECORRPT)+int(DECORRRHO)), 
                     scale=0.0003)(y_hat)
# kin_hats = Adversary(config.n_decorr_bins, n_outputs=2, scale=0.01)(y_hat)

i = [inputs]
pivoter = Model(inputs=i,
                outputs=[y_hat]+kin_hats)
pivoter.compile(optimizer=Adam(lr=0.001),
                loss=['categorical_crossentropy'] + ['categorical_crossentropy' for _ in kin_hats],
                loss_weights=adv_loss_weights)

print '############# ARCHITECTURE #############'
pivoter.summary()
print '###################################'

pred = classifier.predict(test_i)

# ctrl+C now triggers a graceful exit
def save_classifier(name='shallow', model=classifier):
    model.save('models/%s_%s.h5'%(name, APOSTLE))

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


def save_classifier(name='shallow_decorr', model=classifier):
    model.save('models/%s_%s.h5'%(name, APOSTLE))

def save_and_exit(signal=None, frame=None, name='shallow_decorr', model=classifier):
    save_classifier(name, model)
    flog.close()
    exit(1)

signal.signal(signal.SIGINT, save_and_exit)


pivoter.fit_generator(adv_train_gen, 
                         steps_per_epoch=5000, 
                         epochs=NEPOCH*2,
                         validation_data=adv_validation_gen,
                         validation_steps=10,
                        )
save_classifier()
