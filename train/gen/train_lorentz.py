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
from subtlenet.generators.gen_4vec import make_coll, generate, get_dims
from subtlenet.backend.keras_layers import *
from subtlenet.backend.layers import *

''' 
some global definitions
''' 

NEPOCH = 10
APOSTLE = 'v1'
system('cp %s lorentz_models/train_%s.py'%(argv[0], APOSTLE))
config.limit = 50
#config.DEBUG = True

''' 
instantiate data loaders 
''' 
basedir = '/fastscratch/snarayan/genarrays/v_deepgen_1'
top = make_coll(basedir + '/PARTITION/Top_*_CATEGORY.npy')
hig = make_coll(basedir + '/PARTITION/Higgs_*_CATEGORY.npy')
qcd = make_coll(basedir + '/PARTITION/QCD_*_CATEGORY.npy')

data = [top, hig, qcd]

dims0, dims1 = get_dims(data[0])


'''
build the classifier!
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
input_4vec  = Input(shape=(dims0[1], dims0[2]), name='input_4vec')
input_misc  = Input(shape=(dims1[1], dims1[2]), name='input_misc')
input_mass = Input(shape=(1,), name='input_mass')
input_pt = Input(shape=(1,), name='input_pt')
inputs = [input_4vec, input_misc, input_mass, input_pt]

# now build the 4-vector networks
h = LorentzInner(name='lorentz_inner', return_sequences=True)(input_4vec)
inner = CuDNNLSTM(16, name='lstm_inner', return_sequences=True)(h)

h = LorentzOuter(name='lorentz_outer', return_sequences=True)(input_4vec)
outer = CuDNNLSTM(64, name='lstm_outer', return_sequences=True)(h)

# now build the misc network
misc = CuDNNLSTM(32, name='lstm_misc', return_sequences=True)(input_misc)

# now put it all together
h = concatenate([inner, outer, misc], axis=-1)
h = CuDNNLSTM(1024, name='lstm_final', return_sequences=True)(h)
h= Flatten()(h)

h = Dense(256, activation='relu',name='particles_lstm_dense1',kernel_initializer='lecun_uniform')(h)
h = Dropout(0.1)(h)
particles_final = BatchNormalization(momentum=0.6,name='particles_lstm_dense_norm1')(h)
h = Dense(256, activation='relu',name='particles_lstm_dense2',kernel_initializer='lecun_uniform')(h)
h = Dropout(0.1)(h)
particles_final = BatchNormalization(momentum=0.6,name='particles_lstm_dense_norm2')(h)
h = Dense(256, activation='relu',name='particles_lstm_dense3',kernel_initializer='lecun_uniform')(h)
h = Dropout(0.1)(h)
particles_final = BatchNormalization(momentum=0.6,name='particles_lstm_dense_norm3')(h)

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
    model.save('lorentz_models/%s_%s.h5'%(name, APOSTLE))

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
                         callbacks = [ModelCheckpoint('lorentz_models/classifier_conv_%s_{epoch:02d}_{val_loss:.5f}.h5'%APOSTLE)],
                        )
save_classifier()

