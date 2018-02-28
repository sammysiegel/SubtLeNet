#!/usr/local/bin/python2.7

from sys import exit, stdout, argv
from os import environ, system
environ['KERAS_BACKEND'] = 'tensorflow'
environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import signal

from keras.layers import Input, Dense, Dropout, concatenate, LSTM, BatchNormalization, Conv1D, Lambda
from keras.models import Model 
from keras.callbacks import ModelCheckpoint, LambdaCallback, TensorBoard
from keras.optimizers import Adam, SGD
from keras.utils import plot_model
from keras import backend as K

from subtlenet import config 
from subtlenet.generators.gen_singletons import make_coll, generate
from subtlenet.backend.losses import sculpting_kl_penalty
from subtlenet.backend.layers import Adversary
from paths import basedir, figsdir

### some global definitions ### 

NEPOCH = 20
APOSTLE = 'v4_0'
modeldir = 'cce_adversary/'
system('mkdir -p %s'%modeldir)
system('cp %s %s/train_%s.py'%(argv[0], modeldir, APOSTLE))

### instantiate data loaders ### 
top = make_coll(basedir + '/PARTITION/Top_*_CATEGORY.npy')
qcd = make_coll(basedir + '/PARTITION/QCD_*_CATEGORY.npy')

data = [top, qcd]

### first build the classifier! ###

# set up data 
opts = {'decorr_mass':False}
classifier_train_gen = generate(data, partition='train', batch=1000, **opts)
classifier_validation_gen = generate(data, partition='validate', batch=10000, **opts)
classifier_test_gen = generate(data, partition='test', batch=10, **opts)
test_i, test_o, test_w = next(classifier_test_gen)

inputs  = Input(shape=(len(config.gen_default_variables),), name='input')

dense   = Dense(32, activation='tanh',name='dense1',kernel_initializer='lecun_uniform') (inputs)
dense   = Dense(32, activation='tanh',name='dense2',kernel_initializer='lecun_uniform') (dense)
dense   = Dense(32, activation='tanh',name='dense3',kernel_initializer='lecun_uniform') (dense)
y_hat   = Dense(config.n_truth, activation='softmax')                                   (dense)

classifier = Model(inputs=[inputs], outputs=[y_hat])
classifier.compile(optimizer=Adam(lr=0.0005),
                   loss=['categorical_crossentropy'],
                   metrics=['accuracy'])

### now the adversary ###
opts = {'decorr_mass':True}
adversary_train_gen = generate(data, partition='train', batch=1000, **opts)
adversary_validation_gen = generate(data, partition='validate', batch=10000, **opts)
adversary_test_gen = generate(data, partition='test', batch=10, **opts)

kin_hats = Adversary(config.n_decorr_bins, n_outputs=1, scale=0.005)(y_hat)

adversary = Model(inputs=[inputs],
                  outputs=[y_hat]+kin_hats)
adversary.compile(optimizer=Adam(lr=0.00025),
                  loss=['categorical_crossentropy'] + ['categorical_crossentropy' for _ in kin_hats],
                  loss_weights=[0.01] + ([50] * len(kin_hats)))

print '########### CLASSIFIER ############'
adversary.summary()
plot_model(adversary, show_shapes=True, show_layer_names=False, to_file=figsdir+'/cce_adversary.png')
print '###################################'

### now train ###

# ctrl+C now triggers a graceful exit
def save_classifier(name='shallow', model=classifier):
    out = '%s/%s_%s.h5'%(modeldir, name, APOSTLE)
    print 'Saving to',out
    model.save(out)

def save_and_exit(signal=None, frame=None, name='shallow', model=classifier):
    save_classifier(name, model)
    exit(1)

signal.signal(signal.SIGINT, save_and_exit)

classifier.fit_generator(classifier_train_gen, 
                         steps_per_epoch=5000, 
                         epochs=20,
                         validation_data=classifier_validation_gen,
                         validation_steps=10,
                        )
save_classifier(name='baseline')

adversary.fit_generator(adversary_train_gen, 
                        steps_per_epoch=5000, 
                        epochs=20,
                        validation_data=adversary_validation_gen,
                        validation_steps=10,
                        )
save_classifier(name='decorrelated')



