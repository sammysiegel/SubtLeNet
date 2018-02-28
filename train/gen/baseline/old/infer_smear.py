#!/usr/local/bin/python2.7


from sys import exit, argv 
from os import environ, system
environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np

from keras.models import Model, load_model 
from subtlenet import config 
import subtlenet.generators.gen as gen
from paths import basedir
from subtlenet.backend.layers import *
from subtlenet.backend.smear import gauss, CaloSmear 

gen.truncate = int(argv[1])
config.limit = int(argv[2])
name = 'trunc%i_limit%i_best'%(gen.truncate, config.limit)
print 'inferring',name
shallow = load_model('smeared_models/classifier_v4_trunc%i_limit%i_best.h5'%(gen.truncate, config.limit),
                     custom_objects={'DenseBroadcast':DenseBroadcast})

coll = gen.make_coll(basedir + '/PARTITION/*_CATEGORY.npy')

calo = CaloSmear(0, 0.01, 0, lambda x : 1./np.sqrt(x))

msd_norm_factor = 1. / config.max_mass
pt_norm_factor = 1. / (config.max_pt - config.min_pt)
msd_index = config.gen_singletons['msd']
pt_index = config.gen_singletons['pt']

def predict_t(data):
    msd = data['singletons'][:,msd_index] * msd_norm_factor
    pt = (data['singletons'][:,pt_index] - config.min_pt) * pt_norm_factor
    if msd.shape[0] > 0:
        particles = data['particles'][:,:config.limit,:-1]
        particles = calo(particles)[:,:,:gen.truncate]
        r_shallow_t = shallow.predict([particles,msd,pt])[:,config.n_truth-1]
    else:
        r_shallow_t = np.empty((0,))

    return r_shallow_t 

coll.infer(['singletons','particles'], f=predict_t, name='smeared_'+name, partition='test')
