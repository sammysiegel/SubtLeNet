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
from paths import basedir, figsdir

''' 
some global definitions
''' 

NEPOCH = 5
APOSTLE = 'v4_nopt'
system('cp %s shallow_models/train_%s.py'%(argv[0], APOSTLE))

''' 
instantiate data loaders 
''' 
top = make_coll(basedir + '/PARTITION/Top_*_CATEGORY.npy')
qcd = make_coll(basedir + '/PARTITION/QCD_*_CATEGORY.npy')

data = [top, qcd]

'''
first build the classifier!
'''

# set up data 
opts = {'kl_decorr_mass':True}
classifier_train_gen = generate(data, partition='train', batch=1000, **opts)
classifier_validation_gen = generate(data, partition='validate', batch=10000, **opts)
classifier_test_gen = generate(data, partition='test', batch=10, **opts)
test_i, test_o, test_w = next(classifier_test_gen)

inputs  = Input(shape=(len(config.gen_default_variables),), name='input')
kl_mass_input  = Input(shape=(config.n_decorr_bins+1,), name='aux_input')

dense   = Dense(32, activation='tanh',name='dense1',kernel_initializer='lecun_uniform') (inputs)
dense   = Dense(32, activation='tanh',name='dense2',kernel_initializer='lecun_uniform') (dense)
dense   = Dense(32, activation='tanh',name='dense3',kernel_initializer='lecun_uniform') (dense)
y_hat   = Dense(config.n_truth, activation='softmax')                                   (dense)

tag = Lambda(lambda x : x[:,-1:], output_shape=(1,))(y_hat)
kl_output = concatenate([kl_mass_input, tag], axis=-1)

classifier = Model(inputs=[inputs, kl_mass_input], outputs=[y_hat, kl_output])
classifier.compile(optimizer=Adam(lr=0.0005),
                   loss=['categorical_crossentropy', sculpting_kl_penalty],
                   loss_weights=[1,0.1])


# build this model so we can save just the classification part
# weights are shared with classifier
prediction = Model(inputs=[inputs], outputs=[y_hat])
prediction.compile(optimizer=Adam(lr=0.0005), loss='categorical_crossentropy')

print '########### CLASSIFIER ############'
classifier.summary()
plot_model(classifier, show_shapes=True, show_layer_names=False, to_file=figsdir+'/shallow.png')
print '###################################'


# ctrl+C now triggers a graceful exit
def save_classifier(name='shallow', model=prediction):
    model.save('shallow_models/%s_%s.h5'%(name, APOSTLE))

def save_and_exit(signal=None, frame=None, name='shallow', model=prediction):
    save_classifier(name, model)
    exit(1)

signal.signal(signal.SIGINT, save_and_exit)



classifier.fit_generator(classifier_train_gen, 
                         steps_per_epoch=5000, 
                         epochs=NEPOCH,
                         validation_data=classifier_validation_gen,
                         validation_steps=10,
                        )
save_classifier()

