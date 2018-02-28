#!/usr/local/bin/python2.7

from sys import exit 
from os import environ, system
environ['KERAS_BACKEND'] = 'tensorflow'
environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
environ["CUDA_VISIBLE_DEVICES"] = ""


import numpy as np

from subtlenet import config, utils
import akt_config
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
              'shallow', 
#              'dense',
#              'trunc4_limit10_best', 
#              'trunc7_limit10_best', 
#              'trunc4_limit50_best', 
#              'smeared_trunc7_limit50_best', 
#              'baseline_trunc4_limit50_best', 
#              'trunc7_limit50_best', 
#              'dense_trunc7_limit50_best', 
#              'dense_trunc4_limit100_best', 
#              'trunc7_limit100_best', 
#              'trunc4_limit100_best',
              ]

colls = {
    't' : make_coll(paths.basedir + '/PARTITION/Top_*_CATEGORY.npy',categories=components),
    'q' : make_coll(paths.basedir + '/PARTITION/QCD_*_CATEGORY.npy',categories=components),
}


# run DNN
def predict(data,model):
    return data[model]

def access(data, v):
    return data['singletons'][:,config.gen_singletons[v]]

def div(data, num, den):
    return access(data, num) / np.clip(access(data, den), 0.0001, 999)

f_vars = {
    'nprongs' : (lambda x : access(x, 'nprongs'), np.arange(0,10,0.1), r'$N_\mathrm{prongs}$'),
    'tau32' : (lambda x : div(x, 'tau3', 'tau2'), np.arange(0,1.2,0.01), r'$\tau_{32}$'),
#    'tau21' : (lambda x : div(x, 'tau2', 'tau1'), np.arange(0,1.2,0.01), r'$\tau_{21}$'),
    'tau32sd' : (lambda x : div(x, 'tau3sd', 'tau2sd'), np.arange(0,1.2,0.01), r'$\tau_{32}^\mathrm{sd}$'),
#    'tau21sd' : (lambda x : div(x, 'tau2sd', 'tau1sd'), np.arange(0,1.2,0.01), r'$\tau_{21}^\mathrm{sd}$'),
    'partonm' : (lambda x : access(x, 'partonm'), np.arange(0,400,5), 'Parton mass [GeV]'),
    'msd'     : (lambda x : access(x, 'msd'), np.arange(0.,400.,20.), r'$m_\mathrm{SD}$ [GeV]'),
    'pt'        : (lambda x : access(x, 'pt'), np.arange(250.,1000.,50.), r'$p_\mathrm{T}$ [GeV]'),
    'shallow' : (lambda x : x['shallow'], np.arange(0,1.2,0.01), r'Shallow classifier'),
    'shallow_roc' : (lambda x : x['shallow'], np.arange(0,1.2,0.0001), r'Shallow classifier'),
#    'dense' : (lambda x : x['dense'], np.arange(0,1.2,0.01), r'Deep classifier'),
#    'dense_roc' : (lambda x : x['dense'], np.arange(0,1.2,0.0001), r'Deep classifier'),
#    'lstm4_10'  : (lambda x : x['trunc4_limit10_best'], np.arange(0,1.2,0.01), 'C-LSTM (4,10)'),
#    'lstm7_10'  : (lambda x : x['trunc7_limit10_best'], np.arange(0,1.2,0.01), 'C-LSTM (7,10)'),
#    'lstm4_50'  : (lambda x : x['trunc4_limit50_best'], np.arange(0,1.2,0.01), 'C-LSTM (4,50)'),
#    'smeared_lstm7_50'  : (lambda x : x['smeared_trunc7_limit50_best'], np.arange(0,1.2,0.01), 'Smeared C-LSTM (7,50)'),
#    'lstm4_50'  : (lambda x : x['baseline_trunc4_limit50_best'], np.arange(0,1.2,0.01), 'C-LSTM (4,50)'),
#    'lstm7_50'  : (lambda x : x['trunc7_limit50_best'], np.arange(0,1.2,0.01), 'C-LSTM (7,50)'),
#    'dense7_50'  : (lambda x : x['dense_trunc7_limit50_best'], np.arange(0,1.2,0.01), 'Dense (7,50)'),
#    'dense4_100'  : (lambda x : x['dense_trunc4_limit100_best'], np.arange(0,1.2,0.01), 'Dense (4,100)'),
#    'lstm4_100'  : (lambda x : x['trunc4_limit100_best'], np.arange(0,1.2,0.01), 'C-LSTM (4,100)'),
#    'lstm7_100'  : (lambda x : x['trunc7_limit100_best'], np.arange(0,1.2,0.01), 'C-LSTM (4,50)'),
#    'lstm4_10_roc'  : (lambda x : x['trunc4_limit10_best'], np.arange(0,1.2,0.00001), 'C-LSTM (4,10)'),
#    'lstm7_10_roc'  : (lambda x : x['trunc7_limit10_best'], np.arange(0,1.2,0.00001), 'C-LSTM (7,10)'),
#    'lstm4_50_roc'  : (lambda x : x['trunc4_limit50_best'], np.arange(0,1.2,0.00001), 'C-LSTM (4,50)'),
#    'smeared_lstm7_50_roc'  : (lambda x : x['smeared_trunc7_limit50_best'], np.arange(0,1.2,0.00001), 'Smeared C-LSTM (7,50)'),
#    'lstm4_50_roc'  : (lambda x : x['baseline_trunc4_limit50_best'], np.arange(0,1.2,0.00001), 'C-LSTM (4,50)'),
#    'lstm7_50_roc'  : (lambda x : x['trunc7_limit50_best'], np.arange(0,1.2,0.00001), 'C-LSTM (7,50)'),
#    'dense7_50_roc'  : (lambda x : x['dense_trunc7_limit50_best'], np.arange(0,1.2,0.00001), 'Dense (7,50)'),
#    'dense4_100_roc'  : (lambda x : x['dense_trunc4_limit100_best'], np.arange(0,1.2,0.00001), 'Dense (4,100)'),
#    'lstm4_100_roc'  : (lambda x : x['trunc4_limit100_best'], np.arange(0,1.2,0.00001), 'C-LSTM (4,100)'),
#    'lstm7_100_roc'  : (lambda x : x['trunc7_limit100_best'], np.arange(0,1.2,0.00001), 'C-LSTM (7,100)'),
}

roc_vars = {
            'tau32':(r'$\tau_{32}$',0,':'),
            'tau32sd':(r'$\tau_{32}^\mathrm{SD}$',2,':'),
#            'tau21':(r'$\tau_{21}$',2,':'),
#            'tau21sd':(r'$\tau_{21}^\mathrm{SD}$',3,':'),
            'lstm4_10_roc':('C-LSTM (4,10)',4),
            'lstm4_50_roc':('C-LSTM (4,50)',5),
            'smeared_lstm7_50_roc':('Smeared C-LSTM (7,50)',8),
            'lstm4_100_roc':('C-LSTM (4,100)',6),
            'lstm7_50_roc':('C-LSTM (7,50)',7),
            'lstm7_100_roc':('C-LSTM (7,100)',3),
            'shallow_roc':('Shallow',9,'--'),
            'dense_roc':('Deep (7,50)',11,'--'),
            }

order = [
        'tau32',
        'tau32sd',
        'shallow_roc',
        'dense_roc',
        'lstm4_10_roc',
        'smeared_lstm7_50_roc',
        'lstm4_50_roc',
        'lstm7_50_roc',
        'lstm4_100_roc',
        'lstm7_100_roc',
        ]

# unmasked first
hists = {}
for k,v in colls.iteritems():
    hists[k] = v.draw(components=components,
                      f_vars=f_vars,
                      n_batches=n_batches, partition=partition)

for k in hists['t']:
    if 'roc' in k:
        continue
    ht = hists['t'][k]
    hq = hists['q'][k]
    for h in [ht, hq]:
        h.scale()
    p.clear()
    p.add_hist(ht, '3-prong top', 'r')
    p.add_hist(hq, '1-prong QCD', 'k')
    p.plot(output=OUTPUT+k, xlabel=f_vars[k][2])

r.clear()
r.add_vars(hists['t'],           
           hists['q'],
           roc_vars,
           order
           )
r.plot(**{'output':OUTPUT+'roc'})

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
    if 'roc' in k:
        continue
    ht = hists['t'][k]
    hq = hists['q'][k]
    for h in [ht, hq]:
        h.scale()
    p.clear()
    p.add_hist(ht, '3-prong top', 'r')
    p.add_hist(hq, '1-prong QCD', 'k')
    p.plot(output=OUTPUT+'mass_'+k, xlabel=f_vars[k][2])

r.clear()
r.add_vars(hists['t'],           
           hists['q'],
           roc_vars,
           order
           )
r.plot(**{'output':OUTPUT+'mass_roc'})


