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
basedir = environ['BASEDIR']
figsdir = environ['FIGSDIR']

n_batches = 500
partition = 'test'

p = utils.Plotter()
r1 = utils.Roccer(y_range=range(-5,1))
r2 = utils.Roccer(y_range=range(-4,1))

OUTPUT = figsdir + '/' 
system('mkdir -p %s'%OUTPUT)
OUTPUT += 'scan_limit_'

components = [
              'singletons',
              'shallow', 
              'baseline_Adam_4_10',
              'baseline_Adam_4_50',
              'baseline_Adam_4_100',
              ]

colls = {
    't' : make_coll(basedir + '/PARTITION/Top_*_CATEGORY.npy',categories=components),
    'q' : make_coll(basedir + '/PARTITION/QCD_*_CATEGORY.npy',categories=components),
}


# run DNN
def predict(data,model):
    return data[model]

def access(data, v):
    return data['singletons'][:,config.gen_singletons[v]]

def div(data, num, den):
    return access(data, num) / np.clip(access(data, den), 0.00003, 999)

def makebins(lo, hi, w):
    return np.linspace(lo, hi, int((hi-lo)/w))

f_vars = {
    'nprongs' : (lambda x : access(x, 'nprongs'), makebins(0,10,0.1), r'$N_\mathrm{prongs}$'),
    'tau32' : (lambda x : div(x, 'tau3', 'tau2'), makebins(0,1.2,0.01), r'$\tau_{32}$'),
    'tau32sd' : (lambda x : div(x, 'tau3sd', 'tau2sd'), makebins(0,1.2,0.01), r'$\tau_{32}^\mathrm{sd}$'),
    'partonm' : (lambda x : access(x, 'partonm'), makebins(0,400,5), 'Parton mass [GeV]'),
    'msd'     : (lambda x : access(x, 'msd'), makebins(0.,400.,20.), r'$m_\mathrm{SD}$ [GeV]'),
    'pt'        : (lambda x : access(x, 'pt'), makebins(250.,1000.,50.), r'$p_\mathrm{T}$ [GeV]'),
    'shallow' : (lambda x : x['shallow'], makebins(0,1.2,0.01), r'Shallow (no $p_{T}$) classifier'),
    'shallow_roc' : (lambda x : x['shallow'], makebins(0,1.2,0.00003), r'Shallow (no $p_{T}$) classifier'),
    'baseline_Adam_4_10'  : (lambda x : x['baseline_Adam_4_10'], makebins(0,1,0.01), '(4,10)'),
    'baseline_Adam_4_10_roc'  : (lambda x : x['baseline_Adam_4_10'], makebins(0,1,0.00003), '(4,10)'),
    'baseline_Adam_4_50'  : (lambda x : x['baseline_Adam_4_50'], makebins(0,1,0.01), '(4,50)'),
    'baseline_Adam_4_50_roc'  : (lambda x : x['baseline_Adam_4_50'], makebins(0,1,0.00003), '(4,50)'),
    'baseline_Adam_4_100'  : (lambda x : x['baseline_Adam_4_100'], makebins(0,1,0.01), '(4,100)'),
    'baseline_Adam_4_100_roc'  : (lambda x : x['baseline_Adam_4_100'], makebins(0,1,0.00003), '(4,100)'),
}

roc_vars = {
            'tau32':(r'$\tau_{32}$',0,'--'),
            'tau32sd':(r'$\tau_{32}^\mathrm{SD}$',2,'--'),
            'shallow_roc':('Shallow',4,'--'),
            'baseline_Adam_4_10_roc':('C-LSTM (4,10)',7),
            'baseline_Adam_4_50_roc':('C-LSTM (4,50)',9),
            'baseline_Adam_4_100_roc':('C-LSTM (4,100)',10),
            }

order = [
        'tau32',
        'tau32sd',
        'shallow_roc',
#        'baseline_Adam_4_10_roc',
        'baseline_Adam_4_50_roc',
        'baseline_Adam_4_100_roc',
        ]

# unmasked first
hists = {}
for k,v in colls.iteritems():
    hists[k] = v.draw(components=components,
                      f_vars=f_vars,
                      n_batches=n_batches, partition=partition)

for k in hists['t']:
    ht = hists['t'][k]
    hq = hists['q'][k]
    for h in [ht, hq]:
        h.scale()
    if 'roc' in k:
        continue
    p.clear()
    p.add_hist(ht, '3-prong top', 'r')
    p.add_hist(hq, '1-prong QCD', 'k')
    p.plot(output=OUTPUT+k, xlabel=f_vars[k][2])

r1.clear()
r1.add_vars(hists['t'],           
           hists['q'],
           roc_vars,
           order
           )
r1.plot(**{'output':OUTPUT+'roc'})


bkg_hists = {k:v for k,v in hists['q'].iteritems()}

# mask the top mass
def f_mask(data):
    mass = data['singletons'][:,config.gen_singletons['msd']]
    return (mass > 150) & (mass < 200)

hists = {}
for k,v in colls.iteritems():
    hists[k] = v.draw(components=components,
                      f_vars=f_vars,
                      n_batches=n_batches, partition=partition,
                      f_mask=f_mask)

for k in hists['t']:
    ht = hists['t'][k]
    hq = hists['q'][k]
    for h in [ht, hq]:
        h.scale()
    if 'roc' in k:
        continue
    p.clear()
    p.add_hist(ht, '3-prong top', 'r')
    p.add_hist(hq, '1-prong QCD', 'k')
    p.plot(output=OUTPUT+'mass_'+k, xlabel=f_vars[k][2])

r2.clear()
r2.add_vars(hists['t'],           
           hists['q'],
           roc_vars,
           order
           )
r2.plot(**{'output':OUTPUT+'mass_roc'})

