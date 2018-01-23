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
obj.DEBUG = True
# obj.truth = 'resonanceType'
# config.n_truth = 5


n_batches = 500
partition = 'test'


def make_coll(fpath):
    coll = obj.GenCollection()
    coll.add_categories(['singletons'], fpath) 
    return coll 

t = make_coll('/data/t3serv014/snarayan/genarrays/v0_repro/PARTITION/ZprimeToTT_top_*_CATEGORY.npy')
w = make_coll('/data/t3serv014/snarayan/genarrays/v0_repro/PARTITION/ZprimeToWW_w_*_CATEGORY.npy')

qcd_t = make_coll('/data/t3serv014/snarayan/genarrays/v0_repro/PARTITION/QCD_top_*_CATEGORY.npy')
qcd_w = make_coll('/data/t3serv014/snarayan/genarrays/v0_repro/PARTITION/QCD_w_*_CATEGORY.npy')


# run DNN

def f(v, args=(0,1.2,0.01)):
    g = lambda x : x['singletons'][:,obj.gen_singletons[v]]
    return (g, np.arange(*args))

f_vars = { v:f(v) for v in ['t32','t32sd','parton_t32'] }
f_vars['pt'] = f('pt', (400,1000,40))
f_vars['msd'] = f('msd', (0,1000,40))
f_vars['mass'] = f('mass', (0,1000,40))

def f_mask(x):
    lower = x['singletons'][:,obj.gen_singletons['msd']] > 110
    upper = x['singletons'][:,obj.gen_singletons['msd']] < 210
    return (lower & upper)

OUTPUT = '/home/snarayan/public_html/figs/badnet/partons/'
system('mkdir -p '+OUTPUT)

p = utils.Plotter()
r = utils.Roccer()


# unmasked first
hists_top = t.draw(components=['singletons'],
                   f_vars=f_vars, n_batches=n_batches, partition=partition,
                   f_mask=f_mask)
hists_qcd = qcd_t.draw(components=['singletons'],
                       f_vars=f_vars, n_batches=n_batches, partition=partition,
                       f_mask=f_mask)

for k in hists_top:
    htop = hists_top[k]
    hqcd = hists_qcd[k]
    htop.scale() 
    hqcd.scale()
    p.clear()
    p.add_hist(htop, 'top', 'r')
    p.add_hist(hqcd, 'qcd', 'k')
    p.plot({'output':OUTPUT+'unmasked_'+k})

r.clear()
r.add_vars(hists_top,
           hists_qcd,
           {'t32':'Hadrons', 't32sd':'Groomed hadrons',
            'parton_t32':'Partons'},
           )
r.plot({'output':OUTPUT+'unmasked_roc'})

