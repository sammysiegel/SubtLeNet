#!/usr/local/bin/python2.7


from sys import exit, argv 
from os import environ, system
environ['KERAS_BACKEND'] = 'tensorflow'
# environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
# environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

from keras.models import Model, load_model 
from subtlenet import config 
import subtlenet.generators.gen as gen
from paths import basedir
from subtlenet.backend.layers import *

gen.truncate = int(argv[1])
config.limit = int(argv[2])
name = argv[3] 
name += '_trunc%i_limit%i_best'%(gen.truncate, config.limit)
print 'inferring',name
model = load_model('cce_adversary/stack_%s.h5'%(name.replace('_trunc','_v4_trunc')),
                     custom_objects={'DenseBroadcast':DenseBroadcast,
                                     'GradReverseLayer':GradReverseLayer})

model.summary()

coll = gen.make_coll(basedir + '/PARTITION/*_CATEGORY.npy')

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
        r_model_t = model.predict([particles,msd,pt])[0][:,config.n_truth-1]
    else:
        r_model_t = np.empty((0,))

    return r_model_t 

coll.infer(['singletons','particles'], f=predict_t, name=name, partition='test')
