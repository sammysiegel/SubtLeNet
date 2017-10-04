#!/usr/local/bin/python2.7


from sys import exit 
from os import environ, system
environ['KERAS_BACKEND'] = 'tensorflow'
# environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
# environ["CUDA_VISIBLE_DEVICES"] = ""


import numpy as np
import utils
import adversarial


from keras.models import Model, load_model 
from keras.utils import np_utils
import obj 
import config 
# config.DEBUG = True
config.n_truth = 5
config.truth = 'resonanceType'



shallow = load_model('models/tauNmsd.h5')
classifier = load_model('models/classifier.h5')
regularized = load_model('models/regularized.h5')

basedir = '/fastscratch/snarayan/baconarrays/v12_repro/'
system('rm %s/test/*nn.npy'%basedir)
coll = obj.PFSVCollection()
coll.add_categories(['singletons','inclusive'], 
                    basedir+'/PARTITION/*CATEGORY.npy')

def predict_t(data):
    inputs = data['singletons'][:,[obj.singletons['tau32'],obj.singletons['tau21'],obj.singletons['msd']]]
    mus = np.array([0.5, 0.5, 75])
    sigmas = np.array([0.5, 0.5, 50])
    inputs -= mus 
    inputs /= sigmas 
    r_shallow = shallow.predict(inputs)[:,config.n_truth-1]

    r_classifier = classifier.predict([data['inclusive']])[:,config.n_truth-1]
    r_regularized = regularized.predict([data['inclusive']])[:,config.n_truth-1]

    return np.vstack([r_shallow, r_classifier, r_regularized]).T 

coll.infer(['singletons', 'inclusive'], f=predict_t, name='nn', partition='test')
