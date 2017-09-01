#!/usr/local/bin/python2.7

from sys import exit 
from os import environ, system
environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import utils
import signal

from keras.layers import Input, Dense, Dropout, concatenate, LSTM, BatchNormalization
from keras.models import Model 
from keras.callbacks import ModelCheckpoint, LambdaCallback, TensorBoard
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras import backend as K
K.set_image_data_format('channels_last')

from adversarial import Adversary
import obj 
import config 
config.DEBUG = True

''' 
instantiate data loaders 
''' 

def make_coll(fpath):
    coll = obj.PFSVCollection()
    coll.add_categories(['singletons', 'inclusive'], fpath) 
    return coll 

top = make_coll('/home/snarayan/scratch5/baconarrays/v11_repro/PARTITION/ZprimeToTTJet_3_*_CATEGORY.npy')
higgs = make_coll('/data/t3serv014/bmaier/baconarrays/v1_repro//PARTITION/ZprimeToA0hToA0chichihbb_2_*_CATEGORY.npy')
qcd = make_coll('/home/snarayan/scratch5/baconarrays/v11_repro/PARTITION/QCD_1_*_CATEGORY.npy') 

data = [top, higgs, qcd]

# preload some data just to get the dimensions
data[0].objects['train']['inclusive'].load(memory=False)
dims = data[0].objects['train']['inclusive'].data.shape 
dims = (None, 20, 9) # override

''' 
some global definitions
''' 

system('mv train_lstm.log train_lstm.log.old')
flog = open('train_lstm.log','w')
callback = LambdaCallback(
    on_batch_end=lambda batch, logs: flog.write('batch=%i,logs=%s\n'%(batch,str(logs)))
)

tb = TensorBoard(
    log_dir = './lstm_logs',
    write_graph = True,
    write_images = True
)

'''
first build the classifier!
'''

# set up data 
classifier_train_gen = obj.generatePF(data, partition='train', batch=100, normalize=False)
classifier_validation_gen = obj.generatePF(data, partition='validate', batch=100)
classifier_test_gen = obj.generatePF(data, partition='validate', batch=1000)
test_i, test_o, test_w = next(classifier_test_gen)

inputs  = Input(shape=(dims[1], dims[2]), name='input')
norm    = BatchNormalization(momentum=0.6, name='input_bnorm')                              (inputs)
lstm    = LSTM(100, go_backwards=True, implementation=2, name='lstm')                       (norm)
norm    = BatchNormalization(momentum=0.6, name='lstm_norm')                                (lstm)
dense   = Dense(100, activation='relu',name='lstmdense',kernel_initializer='lecun_uniform') (norm)
norm    = BatchNormalization(momentum=0.6,name='lstmdense_norm')                            (dense)
for i in xrange(5):
    dense = Dense(50, activation='relu',name='dense%i'%i)(norm)
    norm = BatchNormalization(momentum=0.6,name='dense%i_norm'%i)(dense)
y_hat   = Dense(config.n_truth, activation='softmax')                                       (norm)

classifier = Model(inputs=inputs, outputs=y_hat)
classifier.compile(optimizer=Adam(lr=0.0005),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

print '########### CLASSIFIER ############'
classifier.summary()
print '###################################'

pred = classifier.predict(test_i)


# ctrl+C now triggers a graceful exit
def save_classifier(name='classifier', model=classifier):
    model.save('%s.h5'%name)

def save_and_exit(signal=None, frame=None, name='classifier', model=classifier):
    save_classifier(name, model)
    flog.close()
    exit(1)

signal.signal(signal.SIGINT, save_and_exit)


''' 
now build the adversarial setup
'''

# set up data 
train_gen = obj.generatePF(data, partition='train', batch=100, decorr_mass=True, normalize=False)
validation_gen = obj.generatePF(data, partition='validate', batch=100, decorr_mass=True)
test_gen = obj.generatePF(data, partition='validate', batch=1000, decorr_mass=True)

# build the model 
mass_hat = Adversary(config.n_mass_bins, scale=0.01)(y_hat)

pivoter = Model(inputs=[inputs],
                outputs=[y_hat, mass_hat])
pivoter.compile(optimizer=Adam(lr=0.001, clipnorm=1., clipvalue=0.5),
                loss=['categorical_crossentropy', 'categorical_crossentropy'],
                loss_weights=[0.1,10])

print '############# PIVOTER #############'
pivoter.summary()
print '###################################'

'''
Now we train both models
'''

# bit of pre-training to get the classifer in the right place 
classifier.fit_generator(classifier_train_gen, 
                         steps_per_epoch=5000, 
                         epochs=5,
                         validation_data=classifier_validation_gen,
                         validation_steps=1000)


save_classifier(name='pretrained')


def save_and_exit(signal=None, frame=None, name='regularized', model=classifier):
    save_classifier(name, model)
    flog.close()
    exit(1)
signal.signal(signal.SIGINT, save_and_exit)

# now train the model for real
pivoter.fit_generator(train_gen, 
                      steps_per_epoch=5000,
                      epochs=8,
                      callbacks=[callback, tb],
                      validation_data=validation_gen,
                      validation_steps=100)


save_classifier(name='regularized')

save_and_exit(name='pivoter', model=pivoter)

