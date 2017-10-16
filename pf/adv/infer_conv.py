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

if __name__ == '__main__':

    #config.n_truth = 5
    #config.truth = 'resonanceType'

    classifier = load_model('models/classifier_conv.h5')
    regularized = load_model('models/regularized_conv.h5')

    basedir = '/fastscratch/snarayan/baconarrays/v13_repro/'
    system('rm -f %s/test/*jesus_conv.npy'%basedir)
    coll = obj.PFSVCollection()
    coll.add_categories(['singletons','inclusive'], 
                        basedir+'/PARTITION/*CATEGORY.npy')

    def predict_t(data):
        r_classifier = classifier.predict([data['inclusive']])
        r_classifier_top = r_classifier[:,config.n_truth-1]
        r_classifier_higgs = r_classifier[:,config.n_truth-2]

        r_regularized = regularized.predict([data['inclusive']])
        r_regularized_top = r_regularized[:,config.n_truth-1]
        r_regularized_higgs = r_regularized[:,config.n_truth-2]

        #return np.vstack([r_classifier, r_regularized]).T 
        return np.vstack([r_classifier_top, r_classifier_higgs, r_regularized_top, r_regularized_higgs]).T 

    coll.infer(['singletons', 'inclusive'], f=predict_t, name='jesus_conv', partition='test', ncores=1)
