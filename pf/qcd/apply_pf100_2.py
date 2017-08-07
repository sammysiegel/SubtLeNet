#!/usr/local/bin/python2.7

from sys import exit 
from os import environ, system
environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import utils


from keras.models import Model, load_model 
from keras.utils import np_utils
import obj 
# obj.DEBUG = True
# obj.truth = 'resonanceType'
# obj.n_truth = 5

model = load_model('model_lstm.h5')
model.summary()

n_batches = 400
partition = 'test'

from keras import backend as K
K.set_image_data_format('channels_last')


def make_coll(fpath):
    coll = obj.PFSVCollection()
    coll.add_categories(['singletons', 'inclusive'], fpath) 
    return coll 

# hig_2 = make_coll('/home/snarayan/hscratch/baconarrays/v8_repro/PARTITION/RSGluonToTT_3_*_CATEGORY.npy') # T
top_3 = make_coll('/home/snarayan/hscratch/baconarrays/v8_repro/PARTITION/RSGluonToTT_3_*_CATEGORY.npy') # T
qcd_3 = make_coll('/home/snarayan/hscratch/baconarrays/v8_repro/PARTITION/QCD_3_*_CATEGORY.npy') # T
qcd_2 = make_coll('/home/snarayan/hscratch/baconarrays/v8_repro/PARTITION/QCD_2_*_CATEGORY.npy') # T
qcd_1 = make_coll('/home/snarayan/hscratch/baconarrays/v8_repro/PARTITION/QCD_1_*_CATEGORY.npy') # T


# run DNN
def predict(data):
    return model.predict(data['inclusive'])[:,obj.n_truth-1]

f_vars = {
  'tau32' : (lambda x : x['singletons'][:,obj.singletons['tau32']], np.arange(0,1.2,0.01)),
  'msd'   : (lambda x : x['singletons'][:,obj.singletons['msd']], np.arange(0.,400.,10.)),
  'pt'    : (lambda x : x['singletons'][:,obj.singletons['pt']], np.arange(250.,1000.,50.)),
  'dnn'   : (predict, np.arange(0,1.2,0.01)),
}

OUTPUT = '/home/snarayan/public_html/figs/testplots/qcd13/'
system('mkdir -p '+OUTPUT)

p = utils.Plotter()
r = utils.Roccer()

# now mask the mass
# def mask(data):
#     lower = data['singletons'][:,obj.singletons['msd']] > 110
#     higher = data['singletons'][:,obj.singletons['msd']] < 210
#     pt = data['singletons'][:,obj.singletons['pt']] > 400 
#     return np.logical_and(pt, np.logical_and(lower,higher))
# 
# hists_top = top_4.draw(components=['singletons', 'inclusive'],
#                        f_vars=f_vars, f_mask=mask, n_batches=n_batches,
#                        partition=partition)
# hists_qcd = qcd_0.draw(components=['singletons', 'inclusive'],
#                        f_vars=f_vars, f_mask=mask, n_batches=n_batches,
#                        partition=partition)
# 
# for k in hists_top:
#     htop = hists_top[k]
#     hqcd = hists_qcd[k]
#     htop.scale() 
#     hqcd.scale()
#     p.clear()
#     p.add_hist(htop, '3-prong', 'r')
#     p.add_hist(hqcd, '1-prong', 'k')
#     p.plot({'output':OUTPUT+'masked_'+k})
# 
# r.clear()
# r.add_vars(hists_top,
#            hists_qcd,
#            {'tau32':r'$\tau_{32}$', 'dnn':'DNN', 'msd':r'$m_{SD}$'},
#            {'tau32':'k', 'dnn':'r', 'msd':'b'})
# r.plot({'output':OUTPUT+'masked_roc'})


# unmasked first
hists_t = top_3.draw(components=['singletons', 'inclusive'],
                     f_vars=f_vars, n_batches=n_batches)
hists_3 = qcd_3.draw(components=['singletons', 'inclusive'],
                     f_vars=f_vars, n_batches=n_batches)
hists_2 = qcd_2.draw(components=['singletons', 'inclusive'],
                     f_vars=f_vars, n_batches=n_batches)
hists_1 = qcd_1.draw(components=['singletons', 'inclusive'],
                     f_vars=f_vars, n_batches=n_batches)

for k in hists_1:
    ht = hists_t[k]
    h1 = hists_1[k]
    h2 = hists_2[k]
    h3 = hists_3[k]
    ht.scale() 
    h1.scale() 
    h2.scale() 
    h3.scale() 
    p.clear()
    p.add_hist(ht, '3-prong Top', 'b')
    p.add_hist(h3, '3-prong QCD', 'r')
    p.add_hist(h2, '2-prong QCD', 'g')
    p.add_hist(h1, '1-prong QCD', 'k')
    p.plot({'output':OUTPUT+'unmasked_'+k})

r.clear()
r.add_vars(hists_3,
           hists_1,
           {'tau32':r'$\tau_{32}$', 'dnn':'DNN', 'msd':r'$m_{SD}$'},
           {'tau32':'k', 'dnn':'r', 'msd':'b'})
r.plot({'output':OUTPUT+'unmasked_roc'})

r.clear()
r.add_vars(hists_t,
           hists_1,
           {'tau32':r'$\tau_{32}$', 'dnn':'DNN', 'msd':r'$m_{SD}$'},
           {'tau32':'k', 'dnn':'r', 'msd':'b'})
r.plot({'output':OUTPUT+'unmasked_top_roc'})