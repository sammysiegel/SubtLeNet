from sys import exit, stdout
import sys
from os import environ, system, getenv
environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import signal

from keras.models import Model, load_model 
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, RMSprop, Adadelta, Adagrad
from keras.utils import np_utils
from keras import backend as K
K.set_image_data_format('channels_last')

from .. import config 
from .. import utils
from ..backend.layers import * 
from ..backend.callbacks import *
from ..backend.losses import *
from ..backend.smear import *
from ..backend.keras_objects import *
from ..backend import keras_objects

# train any model
def base_trainer(MODELDIR, _APOSTLE, NEPOCH, model, name, train_gen, validation_gen, save_clf_params=None):
    if save_clf_params is not None:
        callbacks = [PartialModelCheckpoint(filepath='%s/%s/%s_clf_best.h5'%(MODELDIR,_APOSTLE,name), 
                                            save_best_only=True, verbose=True,
                                            **save_clf_params)]
        save_clf = save_clf_params['partial_model']
    else:
        save_clf = model
        callbacks = []
    callbacks += [ModelCheckpoint('%s/%s/%s_best.h5'%(MODELDIR,_APOSTLE,name), 
                                  save_best_only=True, verbose=True)]

    def save_classifier(name_=name, model_=save_clf):
        model_.save('%s/%s/%s.h5'%(MODELDIR,_APOSTLE,name_))

    def save_and_exit(signal=None, frame=None):
        save_classifier()
        exit(1)

    signal.signal(signal.SIGINT, save_and_exit)

    model.fit_generator(train_gen, 
                        steps_per_epoch=3000, 
                        epochs=NEPOCH,
                        validation_data=validation_gen,
                        validation_steps=100,
                        callbacks = callbacks,
                       )
    save_classifier()

