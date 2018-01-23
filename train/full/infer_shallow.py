#!/usr/local/bin/python2.7


from sys import exit 
from os import environ, system
environ['KERAS_BACKEND'] = 'tensorflow'
environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
environ["CUDA_VISIBLE_DEVICES"] = ""


import numpy as np
import utils
import adversarial


from keras.models import Model, load_model 
from keras.utils import np_utils
import obj 
import config 
# config.DEBUG = True
#APOSTLE = 'panda_3'
APOSTLE = 'panda_5_decorr'

if __name__ == '__main__':

    #config.n_truth = 5
    #config.truth = 'resonanceType'

    classifier = load_model('models/shallow_%s.h5'%APOSTLE)
    regularized = load_model('models/shallow_decorr_%s.h5'%APOSTLE)

    basedir = '/fastscratch/snarayan/pandaarrays/v1/'
    system('rm -f %s/test/*%s_shallow.npy'%(basedir, APOSTLE))
    coll = obj.PFSVCollection()
    coll.add_categories(['singletons'], 
                        basedir+'/PARTITION/*CATEGORY.npy')

    idxs = [obj.singletons[x] for x in obj.default_variables]

    def predict_t(data):
        X = data['singletons'][:,idxs]
        X -= obj.default_mus 
        X /= obj.default_sigmas
        r_classifier = classifier.predict(X)
        r_classifier_top = r_classifier[:,config.n_truth-1]
        r_regularized = regularized.predict(X)
        r_regularized_top = r_regularized[:,config.n_truth-1]
        return np.vstack([r_classifier_top, r_regularized_top]).T

    coll.infer(['singletons'], f=predict_t, name='%s_shallow'%(APOSTLE), partition='test', ncores=1)
