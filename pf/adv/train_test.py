#!/usr/local/bin/python2.7

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

top_4 = make_coll('/home/snarayan/scratch5/baconarrays/v9_repro/PARTITION/ZprimeToTTJet_3_*_CATEGORY.npy') # T
qcd_0 = make_coll('/home/snarayan/scratch5/baconarrays/v9_repro/PARTITION/QCD_1_*_CATEGORY.npy') # T

data = [top_4, qcd_0]


# build the discriminator
inputs = Input(shape=(3,), name='inputs')

dense = Dense(10, activation='tanh', kernel_initializer='lecun_uniform')(inputs)
dense = Dense(10, activation='tanh', kernel_initializer='lecun_uniform')(dense)

category_pred =  Dense(config.n_truth, activation='softmax', kernel_initializer='lecun_uniform')(dense)

NMASSBINS = config.n_mass_bins
mass_pred = adversarial.Adversary(NMASSBINS, scale=0.1)(category_pred)

classifier = Model(inputs=inputs, outputs=category_pred)
classifier.compile(optimizer=Adam(lr=0.001),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])


pivoter = Model(inputs=inputs, outputs=[category_pred, mass_pred])
pivoter.compile(optimizer=Adam(lr=0.0001),
                loss=['categorical_crossentropy', 'categorical_crossentropy'],
                loss_weights=[1,20])


classifier.summary()
pivoter.summary()

train_generator = obj.generateTest(data, partition='train', batch=1000)
test_generator = obj.generateTest(data, partition='test', batch=5, decorr_mass=False)
validation_generator = obj.generateTest(data, partition='validation', batch=500)

classifier_train_generator = obj.generateTest(data, partition='train', batch=1000, decorr_mass=False)

test_i, test_o, test_w = next(test_generator)
pred = classifier.predict(test_i)
print test_i[0]
print test_o[0]
print pred


system('mv train.log train.log.old')
flog = open('train.log','w')
callback = LambdaCallback(
    on_batch_end=lambda batch, logs: flog.write('batch=%i,acc=%f,loss=%f\n'%(batch,logs['acc'],logs['loss']))
    )

tb = TensorBoard(
    log_dir = './logs',
    write_graph = True,
    write_images = True
    )

def save_model(name='classifier'):
    classifier.save('%s.h5'%name)

def save_and_exit(signal=None, frame=None):
    save_model()
    flog.close()
    exit(1)

# ctrl+C now triggers a graceful exit
signal.signal(signal.SIGINT, save_and_exit)


classifier.fit_generator(classifier_train_generator, 
                            steps_per_epoch=1000, 
                            epochs=2)

save_model('pretrained')

pivoter.fit_generator(train_generator,
                      steps_per_epoch=1000,
                      callbacks=[tb],
                      epochs=10)

save_model('regularized')

pred = classifier.predict(test_i)
print test_i[0]
print test_o[0]
print pred

