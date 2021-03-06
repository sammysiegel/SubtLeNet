#!/usr/bin/env python2.7

from sys import exit, stdout, argv
from os import environ, system
environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import signal

from keras.models import Model 
from keras.callbacks import ModelCheckpoint, LambdaCallback, TensorBoard
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras import backend as K
K.set_image_data_format('channels_last')

from subtlenet import config 
from subtlenet.generators.gen import make_coll, generate, get_dims
from subtlenet.backend.keras_layers import *

''' 
some global definitions
''' 

NEPOCH = 10
APOSTLE = 'v1'
system('cp %s lstmseq_models/train_%s.py'%(argv[0], APOSTLE))
config.limit = 50
#config.DEBUG = True

''' 
instantiate data loaders 
''' 
#basedir = '/local/snarayan/genarrays/v_deepgen_0'
basedir = '/fastscratch/snarayan/genarrays/v_deepgen_0'
top = make_coll(basedir + '/PARTITION/Top_*_CATEGORY.npy')
hig = make_coll(basedir + '/PARTITION/Higgs_*_CATEGORY.npy')
qcd = make_coll(basedir + '/PARTITION/QCD_*_CATEGORY.npy')

data = [top, hig, qcd]

dims = get_dims(data[0])

'''
first build the classifier!
'''

# set up data 
opts = {
        'learn_mass' : True,
        'learn_pt' : True,
        }
classifier_train_gen = generate(data, partition='train', batch=500, **opts)
classifier_validation_gen = generate(data, partition='validate', batch=1000, **opts)
classifier_test_gen = generate(data, partition='test', batch=10, **opts)
test_i, test_o, test_w = next(classifier_test_gen)

# build all inputs
input_particles  = Input(shape=(dims[1], dims[2]), name='input_particles')
input_mass = Input(shape=(1,), name='input_mass')
input_pt = Input(shape=(1,), name='input_pt')
inputs = [input_particles, input_mass, input_pt]

# now build the particle network
h = BatchNormalization(momentum=0.6, name='particles_input_bnorm')(input_particles)
h = Conv1D(32, 2, activation='relu', name='particles_conv0', kernel_initializer='lecun_uniform', padding='same')(h)
h = BatchNormalization(momentum=0.6, name='particles_conv0_bnorm')(h)
h = Conv1D(16, 4, activation='relu', name='particles_conv1', kernel_initializer='lecun_uniform', padding='same')(h)
particles_conv = BatchNormalization(momentum=0.6, name='particles_conv1_bnorm')(h)

h = CuDNNLSTM(32, name='particles_lstm_seq', return_sequences=True)(particles_conv)
particles_lstm_seq = Flatten()(h)

particles_lstm = CuDNNLSTM(128, name='particles_lstm')(particles_conv)

h = concatenate([particles_lstm_seq, particles_lstm])
h = Dense(256, activation='relu',name='particles_lstm_seq_dense1',kernel_initializer='lecun_uniform')(h)
h = Dropout(0.1)(h)
particles_final = BatchNormalization(momentum=0.6,name='particles_lstm_seq_dense_norm1')(h)
h = Dense(256, activation='relu',name='particles_lstm_seq_dense2',kernel_initializer='lecun_uniform')(h)
h = Dropout(0.1)(h)
particles_final = BatchNormalization(momentum=0.6,name='particles_lstm_seq_dense_norm2')(h)
h = Dense(256, activation='relu',name='particles_lstm_seq_dense3',kernel_initializer='lecun_uniform')(h)
h = Dropout(0.1)(h)
particles_final = BatchNormalization(momentum=0.6,name='particles_lstm_seq_dense_norm3')(h)

# merge everything
to_merge = [particles_final, input_mass, input_pt]
h = concatenate(to_merge)

for i in xrange(1,5):
    h = Dense(64, activation='relu',name='final_dense%i'%i)(h)
    if i%2:
        h = Dropout(0.1)(h)
    h = BatchNormalization(momentum=0.6,name='final_dense%i_norm'%i)(h)


y_hat = Dense(config.n_truth, activation='softmax')(h)

classifier = Model(inputs=inputs, outputs=[y_hat])
classifier.compile(optimizer=Adam(lr=0.001),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

print '########### CLASSIFIER ############'
classifier.summary()
print '###################################'


# ctrl+C now triggers a graceful exit
def save_classifier(name='classifier', model=classifier):
    model.save('lstmseq_models/%s_%s.h5'%(name, APOSTLE))

def save_and_exit(signal=None, frame=None, name='classifier', model=classifier):
    save_classifier(name, model)
    flog.close()
    exit(1)

signal.signal(signal.SIGINT, save_and_exit)

classifier.fit_generator(classifier_train_gen, 
                         steps_per_epoch=3000, 
                         epochs=NEPOCH,
                         validation_data=classifier_validation_gen,
                         validation_steps=1000,
                         callbacks = [ModelCheckpoint('lstmseq_models/classifier_conv_%s_{epoch:02d}_{val_loss:.5f}.h5'%APOSTLE)],
                        )
save_classifier()

