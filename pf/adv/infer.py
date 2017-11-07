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
#config.n_truth = 5
#config.truth = 'resonanceType'




shallow = load_model('models/tauNmsd.h5')
#classifier = load_model('models/classifier.h5')
#regularized = load_model('models/regularized.h5')

#top = make_coll('/home/snarayan/scratch5/baconarrays/v13_repro/PARTITION/ZprimeToTTJet_3_*_CATEGORY.npy')
#higgs = make_coll('/home/snarayan/scratch5/baconarrays/v13_repro/PARTITION/ZprimeToA0hToA0chichihbb_2_*_CATEGORY.npy')
#qcd = make_coll('/home/snarayan/scratch5/baconarrays/v13_repro/PARTITION/QCD_1_*_CATEGORY.npy')

basedir = '/home/snarayan/scratch5/baconarrays/v13_repro/'
system('rm %s/test/*shallow.npy'%basedir)
coll = obj.PFSVCollection()
coll.add_categories(['singletons','inclusive'], 
                    basedir+'/PARTITION/*CATEGORY.npy')

def predict_t(data):
    inputs = data['singletons'][:,[obj.singletons['tau32'],obj.singletons['tau21'],obj.singletons['msd']]]
    mus = np.array([0.5, 0.5, 75])
    sigmas = np.array([0.5, 0.5, 50])
    inputs -= mus 
    inputs /= sigmas 
    r_shallow_t = shallow.predict(inputs)[:,config.n_truth-1]
    r_shallow_h = shallow.predict(inputs)[:,config.n_truth-2]

    return np.vstack([r_shallow_t,r_shallow_h]).T 

coll.infer(['singletons', 'inclusive'], f=predict_t, name='shallow', partition='test')
