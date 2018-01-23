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
obj.limit = 20
config.DEBUG = True
APOSTLE = 'panda_4b_akt'

if __name__ == '__main__':

    #config.n_truth = 5
    #config.truth = 'resonanceType'

    #classifier = load_model('models/classifier_conv_%s.h5'%APOSTLE)
    regularized = load_model('models/regularized_conv_%s.h5'%APOSTLE)

    basedir = '/fastscratch/snarayan/pandaarrays/v1_akt/'
    system('rm -f %s/test/*%s_conv.npy'%(basedir, APOSTLE))
    coll = obj.PFSVCollection()
    coll.add_categories(['singletons','pf'], 
                        basedir+'/PARTITION/*CATEGORY.npy')

    def predict_t(data):
        msd_idx = obj.singletons['msd']
        rho_idx = obj.singletons['rawrho']
        pt_idx = obj.singletons['rawpt']
        if obj.limit:
            inclusive = data['pf'][:,:obj.limit,:]
        else:
            inclusive = data['pf']
        msd = data['singletons'][:,msd_idx] / config.max_mass 
        rho = (data['singletons'][:,rho_idx]  - config.min_rho) / config.max_rho
        pt = (data['singletons'][:,pt_idx] - config.min_pt) / config.max_pt
        if pt.shape[0] > 0:
            X = [inclusive, msd, pt]
            try:
               # r_classifier = classifier.predict(X)
               # r_classifier_top = r_classifier[:,config.n_truth-1]
                #return r_classifier_top

                r_regularized = regularized.predict(X)
                r_regularized_top = r_regularized[:,config.n_truth-1]
                return r_regularized_top

                #return np.vstack([r_classifier, r_regularized]).T 
                #return np.vstack([r_classifier_top, r_regularized_top]).T 
            except Exception as e:
                raise e
        else:
            return np.zeros((0,))
    

    coll.infer(['singletons', 'pf'], f=predict_t, name='%s_conv'%(APOSTLE), partition='test', ncores=1)
