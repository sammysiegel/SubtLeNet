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
APOSTLE = 'luke'

if __name__ == '__main__':

    #config.n_truth = 5
    #config.truth = 'resonanceType'

    classifier = load_model('models/classifier_conv_%s.h5'%APOSTLE)
    regularized = load_model('models/regularized_conv_%s.h5'%APOSTLE)

    basedir = '/fastscratch/snarayan/baconarrays/v13_repro/'
    system('rm -f %s/test/*%s_conv.npy'%(basedir, APOSTLE))
    coll = obj.PFSVCollection()
    coll.add_categories(['singletons','inclusive'], 
                        basedir+'/PARTITION/*CATEGORY.npy')

    def predict_t(data):
        msd_idx = obj.singletons['msd']
        r_classifier = classifier.predict([data['inclusive'], data['singletons'][:,msd_idx] / config.max_mass])
        r_classifier_top = r_classifier[:,config.n_truth-1]
        r_classifier_higgs = r_classifier[:,config.n_truth-2]

        r_regularized = regularized.predict([data['inclusive'], data['singletons'][:,msd_idx] / config.max_mass])
        r_regularized_top = r_regularized[:,config.n_truth-1]
        r_regularized_higgs = r_regularized[:,config.n_truth-2]

        #return np.vstack([r_classifier, r_regularized]).T 
        return np.vstack([r_classifier_top, r_classifier_higgs, r_regularized_top, r_regularized_higgs]).T 

    coll.infer(['singletons', 'inclusive'], f=predict_t, name='%s_conv'%(APOSTLE), partition='test', ncores=1)
