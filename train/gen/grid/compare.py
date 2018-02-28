#!/usr/local/bin/python2.7

from sys import exit 
from os import environ, system
environ['KERAS_BACKEND'] = 'tensorflow'
environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
environ["CUDA_VISIBLE_DEVICES"] = ""


import numpy as np

from subtlenet import config, utils
from subtlenet.backend import obj 
from subtlenet.generators.gen import make_coll
import paths 

n_batches = 500
partition = 'test'

p = utils.Plotter()
r = utils.Roccer()

OUTPUT = paths.figsdir + '/' 
system('mkdir -p %s'%OUTPUT)

components = [
              'singletons',
              'shallow_nopt', 
              'trunc4_limit50_best', 
              'trunc7_limit100_best', 
              ]
components_gen = [
              'singletons',
              'shallow_nopt', 
              'baseline_trunc4_limit50_best', 
              'trunc7_limit100_best', 
              ]

basedir = '/data/t3serv014/snarayan/deep/v_deepgen_4_0p02_small/'
basedir_gen = '/fastscratch/snarayan/genarrays/v_deepgen_4_small/'


colls = {
    't' : make_coll(basedir + '/PARTITION/Top_*_CATEGORY.npy',categories=components),
    'q' : make_coll(basedir + '/PARTITION/QCD_*_CATEGORY.npy',categories=components),
}

colls_gen = {
    't' : make_coll(basedir_gen + '/PARTITION/Top_*_CATEGORY.npy',categories=components_gen),
    'q' : make_coll(basedir_gen + '/PARTITION/QCD_*_CATEGORY.npy',categories=components_gen),
}


# run DNN
def predict(data,model):
    return data[model]

def access(data, v):
    return data['singletons'][:,config.gen_singletons[v]]

def div(data, num, den):
    return access(data, num) / np.clip(access(data, den), 0.0001, 999)

f_vars = {
    'tau32' : (lambda x : div(x, 'tau3', 'tau2'), np.arange(0,1.2,0.01), r'$\tau_{32}$'),
    'tau32sd' : (lambda x : div(x, 'tau3sd', 'tau2sd'), np.arange(0,1.2,0.01), r'$\tau_{32}^\mathrm{sd}$'),
    'shallow_nopt_roc' : (lambda x : x['shallow_nopt'], np.arange(0,1.2,0.0001), r'Shallow (no $p_{T}$) classifier'),
    'lstm4_50_roc'  : (lambda x : x['trunc4_limit50_best'], np.arange(0,1.2,0.00001), 'LSTM (4,50)'),
    'lstm7_100_roc'  : (lambda x : x['trunc7_limit100_best'], np.arange(0,1.2,0.00001), 'LSTM (7,100)'),
}

f_vars_gen = {
    'gen_shallow_nopt' : (lambda x : x['shallow_nopt'], np.arange(0,1.2,0.01), r'Shallow (no $p_{T}$) classifier'),
    'gen_shallow_nopt_roc' : (lambda x : x['shallow_nopt'], np.arange(0,1.2,0.0001), r'Shallow (no $p_{T}$) classifier'),
    'gen_lstm4_50_roc'  : (lambda x : x['baseline_trunc4_limit50_best'], np.arange(0,1.2,0.00001), 'LSTM (4,50)'),
    'gen_lstm7_100_roc'  : (lambda x : x['trunc7_limit100_best'], np.arange(0,1.2,0.00001), 'LSTM (7,100)'),
}

roc_vars = {
            'tau32':(r'$\tau_{32}$',0,':'),
            'tau32sd':(r'$\tau_{32}^\mathrm{SD}$',2,':'),
            'lstm4_50_roc':(r'(4,50) $\delta R=0.02$',5,'--'),
            'lstm7_100_roc':(r'(7,100) $\delta R=0.02$',3,'--'),
            'gen_lstm4_50_roc':('(4,50)',5),
            'gen_lstm7_100_roc':('(7,100)',3),
            'gen_shallow_nopt_roc':('Shallow',9,':'),
            }

order = [
        'tau32',
        'tau32sd',
        'gen_shallow_nopt_roc',
        'lstm4_50_roc',
        'lstm7_100_roc',
        'gen_lstm4_50_roc',
        'gen_lstm7_100_roc',
        ]

# unmasked first
hists = {}
for k,v in colls.iteritems():
    hists[k] = v.draw(components=components,
                      f_vars=f_vars,
                      n_batches=n_batches, partition=partition)

for k,v in colls_gen.iteritems():
    hists[k].update(v.draw(components=components_gen,
                           f_vars=f_vars_gen,
                           n_batches=n_batches, partition=partition))

for k in hists['t']:
    if 'roc' in k:
        continue
    ht = hists['t'][k]
    hq = hists['q'][k]
    for h in [ht, hq]:
        h.scale()

r.clear()
r.add_vars(hists['t'],           
           hists['q'],
           roc_vars,
           order
           )
r.plot(**{'output':OUTPUT+'roc'})

def f_mask(data):
    mass = data['singletons'][:,config.gen_singletons['msd']]
    return (mass > 150) & (mass < 200)

hists = {}
for k,v in colls.iteritems():
    hists[k] = v.draw(components=components,
                      f_vars=f_vars,
                      n_batches=n_batches, partition=partition,
                      f_mask=f_mask)

for k,v in colls_gen.iteritems():
    hists[k].update(v.draw(components=components_gen,
                           f_vars=f_vars_gen,
                           n_batches=n_batches, partition=partition))

for k in hists['t']:
    if 'roc' in k:
        continue
    ht = hists['t'][k]
    hq = hists['q'][k]
    for h in [ht, hq]:
        h.scale()

r.clear()
r.add_vars(hists['t'],           
           hists['q'],
           roc_vars,
           order
           )
r.plot(**{'output':OUTPUT+'mass_roc'})

