#!/usr/local/bin/python2.7

from sys import exit, stdout, argv
from os import environ, system, path
environ['KERAS_BACKEND'] = 'tensorflow'
# environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
# environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import signal

from keras.models import Model, load_model 
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K

from subtlenet import config 
from subtlenet.generators.gen import make_coll, generate, get_dims
import subtlenet.generators.gen as generator
from subtlenet.backend.losses import sculpting_kl_penalty
from subtlenet.backend.layers import Adversary
from subtlenet.backend.keras_layers import *
from subtlenet.backend.callbacks import ModelCheckpoint, PartialModelCheckpoint
from paths import basedir, figsdir

### some global definitions ### 

NEPOCH = 20
TRAIN_BASELINE = False
generator.truncate = int(argv[1])
config.limit = int(argv[2])
config.bin_decorr = False
APOSTLE = 'v4_trunc%i_limit%i'%(generator.truncate, config.limit)
modeldir = 'mse_adversary/'
system('mkdir -p %s'%modeldir)
system('cp %s %s/train_%s.py'%(argv[0], modeldir, APOSTLE))

### instantiate data loaders ### 
top = make_coll(basedir + '/PARTITION/Top_*_CATEGORY.npy')
qcd = make_coll(basedir + '/PARTITION/QCD_*_CATEGORY.npy')

data = [top, qcd]
dims = get_dims(top)

### first build the classifier! ###

# set up data 
opts = {
        'learn_mass' : True,
        'learn_pt' : True,
        'decorr_mass':False
       }
classifier_train_gen = generate(data, partition='train', batch=1000, **opts)
classifier_validation_gen = generate(data, partition='validate', batch=10000, **opts)
classifier_test_gen = generate(data, partition='test', batch=10, **opts)
test_i, test_o, test_w = next(classifier_test_gen)

# build all inputs
input_particles  = Input(shape=(dims[1], dims[2]), name='input_particles')
input_mass = Input(shape=(1,), name='input_mass')
input_pt = Input(shape=(1,), name='input_pt')
inputs = [input_particles, input_mass, input_pt]

# now build the particle network
h = BatchNormalization(momentum=0.6)(input_particles)
h = Conv1D(32, 2, activation='relu', kernel_initializer='lecun_uniform', padding='same')(h)
h = BatchNormalization(momentum=0.6)(h)
h = Conv1D(16, 4, activation='relu', kernel_initializer='lecun_uniform', padding='same')(h)
h = BatchNormalization(momentum=0.6)(h)
h = CuDNNLSTM(100)(h)
#h = Dropout(0.1)(h)
h = BatchNormalization(momentum=0.6)(h)
h = Dense(100, activation='relu', kernel_initializer='lecun_uniform')(h)
particles_final = BatchNormalization(momentum=0.6)(h)

# merge everything
to_merge = [particles_final, input_mass, input_pt]
h = concatenate(to_merge)

for i in xrange(1,5):
    if i == 4:
        h = Dense(50, activation='tanh')(h)
    else:
        h = Dense(50, activation='relu')(h)
    h = BatchNormalization(momentum=0.6)(h)


y_hat = Dense(config.n_truth, activation='softmax')(h)

classifier = Model(inputs=inputs, outputs=[y_hat])
classifier.compile(optimizer=Adam(lr=0.0005),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])


### now the adversary ###
opts['decorr_mass'] = True
adversary_train_gen = generate(data, partition='train', batch=1000, **opts)
adversary_validation_gen = generate(data, partition='validate', batch=10000, **opts)
adversary_test_gen = generate(data, partition='test', batch=10, **opts)

kin_hats = Adversary(config.n_decorr_bins, n_outputs=1, scale=0.5)(y_hat)

adversary = Model(inputs=inputs,
                  outputs=[y_hat]+kin_hats)
adversary.compile(optimizer=Adam(lr=0.00025),
                  loss=['categorical_crossentropy'] + ['mean_squared_error' for _ in kin_hats],
                  loss_weights=[0.05] + ([100] * len(kin_hats)))

print '########### CLASSIFIER ############'
adversary.summary()
plot_model(adversary, show_shapes=True, show_layer_names=False, to_file=figsdir+'/mse_adversary.png')
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

if (not TRAIN_BASELINE) and path.isfile('%s/%s_%s_best.h5'%(modeldir,'baseline',APOSTLE)):
    tmp_ = load_model('%s/%s_%s_best.h5'%(modeldir,'baseline',APOSTLE))
    classifier.set_weights(tmp_.get_weights())
else:
    classifier.fit_generator(classifier_train_gen, 
                             steps_per_epoch=3000, 
                             epochs=5,
                             validation_data=classifier_validation_gen,
                             validation_steps=100,
                             callbacks = [ModelCheckpoint('%s/%s_%s_best.h5'%(modeldir,'baseline',APOSTLE), 
                                                          save_best_only=True, 
                                                          verbose=True)],
                            )
    save_classifier(name='baseline')

def save_and_exit(signal=None, frame=None, name='decorrelated', model=classifier):
    save_classifier(name, model)
    exit(1)


adversary.fit_generator(adversary_train_gen, 
                        steps_per_epoch=3000, 
                        epochs=NEPOCH,
                        validation_data=adversary_validation_gen,
                        validation_steps=100,
                        callbacks = [PartialModelCheckpoint(classifier,
                                                            '%s/%s_%s_best.h5'%(modeldir,'decorrelated',APOSTLE), 
                                                            save_best_only=True, 
                                                            verbose=True),
                                     ModelCheckpoint('%s/stack_%s_%s_best.h5'%(modeldir,'decorrelated',APOSTLE), 
                                                     save_best_only=True, 
                                                     verbose=True)],
                        )
save_classifier(name='decorrelated')


# def save_and_exit(signal=None, frame=None, name='shallow', model=classifier):
#     save_classifier(name, model)
#     exit(1)
# 
# classifier.fit_generator(classifier_train_gen, 
#                          steps_per_epoch=3000, 
#                          epochs=NEPOCH,
#                          validation_data=classifier_validation_gen,
#                          validation_steps=100,
#                          callbacks = [ModelCheckpoint('%s/%s_%s_best.h5'%(modeldir,'baseline',APOSTLE), 
#                                                       save_best_only=True, 
#                                                       verbose=True)],
#                         )
# save_classifier(name='baseline')
