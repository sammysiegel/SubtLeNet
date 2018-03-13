#!/usr/local/bin/python2.7

from sys import exit 
from os import environ, system
environ['KERAS_BACKEND'] = 'tensorflow'
environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
environ["CUDA_VISIBLE_DEVICES"] = ""


import numpy as np

import extra_vars
from subtlenet import config, utils
from subtlenet.backend import obj 
from subtlenet.generators.gen import make_coll

n_batches = 500
partition = 'test'

p = utils.Plotter()
r1 = utils.Roccer(y_range=range(-5,1))
r2 = utils.Roccer(y_range=range(-4,1))

OUTPUT = '/home/snarayan/public_html/figs/deepgen/v4_compare/'
system('mkdir -p %s'%OUTPUT)

components = [
              'singletons',
              'shallow', 
              'baseline_7_100',
              ]

components_base = map(lambda x : x.replace('baseline','baseline_Adam'), components)

basedir = {
#            'base' : '/data/t3serv014/snarayan/deep/v_deepgen_4_eta5_small/',
            '0p02' : '/local/snarayan/genarrays/v_deepgen_4_noetaphi_small/',
#            '0p001' : '/local/snarayan/genarrays/v_deepgen_4_finegrid_small/',
        }
basedir_base = '/fastscratch/snarayan/genarrays/v_deepgen_4_small/'


colls = {}
for d,b in basedir.iteritems():
    colls[d] = {
        't' : make_coll(b + '/PARTITION/Top_*_CATEGORY.npy',categories=components),
        'q' : make_coll(b + '/PARTITION/QCD_*_CATEGORY.npy',categories=components),
    }

colls_base = {
        't' : make_coll(basedir_base + '/PARTITION/Top_*_CATEGORY.npy',categories=components_base),
        'q' : make_coll(basedir_base + '/PARTITION/QCD_*_CATEGORY.npy',categories=components_base),
    }


# run DNN
def predict(data,model):
    return data[model]

def access(data, v):
    return data['singletons'][:,config.gen_singletons[v]]

def div(data, num, den):
    return access(data, num) / np.clip(access(data, den), 0.00003, 999)

f_vars = {
    'tau32' : (lambda x : div(x, 'tau3', 'tau2'), np.arange(0,1.2,0.01), r'$\tau_{32}$'),
    'tau32sd' : (lambda x : div(x, 'tau3sd', 'tau2sd'), np.arange(0,1.2,0.01), r'$\tau_{32}^\mathrm{sd}$'),
    'shallow_roc' : (lambda x : x['shallow'], np.arange(0,1.2,0.00003), r'Shallow'),
    'baseline_7_100_roc' : (lambda x : x['baseline_7_100'], np.arange(0,1.2,0.00003), r'C-LSTM'),
}

f_vars_base = {
    'base_tau32' : (lambda x : div(x, 'tau3', 'tau2'), np.arange(0,1.2,0.01), r'$\tau_{32}$'),
    'base_tau32sd' : (lambda x : div(x, 'tau3sd', 'tau2sd'), np.arange(0,1.2,0.01), r'$\tau_{32}^\mathrm{sd}$'),
    'base_shallow_roc' : (lambda x : x['shallow'], np.arange(0,1.2,0.00003), r'Shallow'),
    'base_baseline_7_100_roc' : (lambda x : x['baseline_Adam_7_100'], np.arange(0,1.2,0.00003), r'C-LSTM'),
}

roc_vars = {
            '0p02_tau32sd':(r'$\tau_{32}^\mathrm{SD}$ $\delta R=0.02$',2,':'),
#            '0p001_tau32sd':(r'$\tau_{32}^\mathrm{SD} ~\delta R = 0.001$ ',2,'--'),
            'base_tau32sd':(r'$\tau_{32}^\mathrm{SD}$',2),
            '0p02_shallow_roc':(r'Shallow $\delta R=0.02$',4,':'),
#            '0p001_shallow_roc':(r'(7,100) $\delta R = 0.001$',4,'--'),
            'base_shallow_roc':(r'Shallow',4),
            '0p02_baseline_7_100_roc':(r'(7,100) $\delta R=0.02$',3,':'),
#            '0p001_baseline_7_100_roc':(r'(7,100) $\delta R = 0.001$',3,'--'),
            'base_baseline_7_100_roc':(r'(7,100)',3),
            }

order = [
        '0p02_tau32sd',
#        '0p001_tau32sd',
        'base_tau32sd',
        '0p02_shallow_roc',
#        '0p001_shallow_roc',
        'base_shallow_roc',
        '0p02_baseline_7_100_roc',
#        '0p001_baseline_7_100_roc',
        'base_baseline_7_100_roc',
        ]

# unmasked first
hists = {'t':{}, 'q':{}}
for base,coll in colls.iteritems():
    for k,v in coll.iteritems():
        h_ = v.draw(components=components,
                    f_vars=f_vars,
                    n_batches=n_batches, partition=partition)
        for k_,v_ in h_.iteritems():
            hists[k][base+'_'+k_] = v_
for k,v in colls_base.iteritems():
    h_ = v.draw(components=components_base,
                f_vars=f_vars_base,
                n_batches=n_batches, partition=partition)
    for k_,v_ in h_.iteritems():
        hists[k][k_] = v_

for k in hists['t']:
    ht = hists['t'][k]
    hq = hists['q'][k]
    for h in [ht, hq]:
        h.scale()

r1.clear()
r1.add_vars(hists['t'],           
           hists['q'],
           roc_vars,
           order
           )
r1.plot(**{'output':OUTPUT+'roc'})

# mask the top mass
def f_mask(data):
    mass = data['singletons'][:,config.gen_singletons['msd']]
    return (mass > 150) & (mass < 200)

hists = {'t':{}, 'q':{}}
for base,coll in colls.iteritems():
    for k,v in coll.iteritems():
        h_ = v.draw(components=components,
                    f_vars=f_vars,
                    f_mask=f_mask,
                    n_batches=n_batches, partition=partition)
        for k_,v_ in h_.iteritems():
            hists[k][base+'_'+k_] = v_
for k,v in colls_base.iteritems():
    h_ = v.draw(components=components_base,
                f_vars=f_vars_base,
                f_mask=f_mask,
                n_batches=n_batches, partition=partition)
    for k_,v_ in h_.iteritems():
        hists[k][k_] = v_

for k in hists['t']:
    ht = hists['t'][k]
    hq = hists['q'][k]
    for h in [ht, hq]:
        h.scale()


r2.clear()
r2.add_vars(hists['t'],           
           hists['q'],
           roc_vars,
           order
           )
r2.plot(**{'output':OUTPUT+'mass_roc'})


