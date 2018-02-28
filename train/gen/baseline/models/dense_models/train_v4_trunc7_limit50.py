#!/usr/local/bin/python2.7

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
import subtlenet.generators.gen as generator
from subtlenet.backend.keras_layers import *
from subtlenet.backend.layers import *
from paths import basedir

''' 
some global definitions
''' 

NEPOCH = 20
config.limit = 50
generator.truncate = 7
APOSTLE = 'v4_trunc%i_limit%i'%(generator.truncate, config.limit)
system('cp %s dense_models/train_%s.py'%(argv[0], APOSTLE))

''' 
instantiate data loaders 
''' 
top = make_coll(basedir + '/PARTITION/Top_*_CATEGORY.npy')
qcd = make_coll(basedir + '/PARTITION/QCD_*_CATEGORY.npy')

data = [top, qcd]

dims = get_dims(top)

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
h = DenseBroadcast(150, activation='linear')(input_particles)
h = BatchNormalization(momentum=0.6)(h)
h = Flatten()(h)
h = Dense(1000, activation='relu')(h)
h = Dropout(0.1)(h)
h = Dense(500, activation='relu')(h)
h = Dropout(0.1)(h)
particles_final = BatchNormalization(momentum=0.6)(h)

# merge everything
to_merge = [particles_final, input_mass, input_pt]
h = concatenate(to_merge)

for i in xrange(1,5):
    if i == 4:
        h = Dense(50, activation='tanh')(h)
    else:
        h = Dense(50, activation='relu')(h)
    if i%2:
        h = Dropout(0.1)(h)
    h = BatchNormalization(momentum=0.6)(h)


y_hat = Dense(config.n_truth, activation='softmax')(h)

classifier = Model(inputs=inputs, outputs=[y_hat])
classifier.compile(optimizer=Adam(lr=0.0005),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

print '########### CLASSIFIER ############'
classifier.summary()
print '###################################'


# ctrl+C now triggers a graceful exit
def save_classifier(name='classifier', model=classifier):
    model.save('dense_models/%s_%s.h5'%(name, APOSTLE))

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
                         callbacks = [ModelCheckpoint('dense_models/%s_%s_best.h5'%('classifier',APOSTLE),save_best_only=True, verbose=True)],
                        )
save_classifier()

