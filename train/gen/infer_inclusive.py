#!/usr/local/bin/python2.7


from sys import exit, argv 
from os import environ, system
environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np

from keras.models import Model, load_model 
from subtlenet import config 
from subtlenet.generators.gen import make_coll
import subtlenet.generators.gen as gen
from paths import basedir

gen.truncate = 7
name = argv[1]
shallow = load_model(argv[2])

config.limit = 50
system('rm %s/test/*%s.npy'%(basedir,name))
coll = make_coll(basedir + '/PARTITION/*_CATEGORY.npy')

msd_norm_factor = 1. / config.max_mass
pt_norm_factor = 1. / (config.max_pt - config.min_pt)
msd_index = config.gen_singletons['msd']
pt_index = config.gen_singletons['pt']

def predict_t(data):
    msd = data['singletons'][:,msd_index] * msd_norm_factor
    pt = (data['singletons'][:,pt_index] - config.min_pt) * pt_norm_factor
    if msd.shape[0] > 0:
        if config.limit:
            particles = data['particles'][:,:config.limit,:gen.truncate]
        else:
            particles = data['particles'][:,:,:gen.truncate]
        r_shallow_t = shallow.predict([particles,msd,pt])[:,config.n_truth-1]
    else:
        r_shallow_t = np.empty((0,1))

    return r_shallow_t 

coll.infer(['singletons','particles'], f=predict_t, name=name, partition='test')
