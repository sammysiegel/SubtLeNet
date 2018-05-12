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
r = utils.Roccer(y_range=range(-4,1))

OUTPUT = figsdir + '/' 
system('mkdir -p %s'%OUTPUT)

components = [
              'singletons',
              'shallow', 
#               'baseline_trunc4_limit50_clf_best', 
#               'decorrelated_trunc4_limit50_clf_best', 
#               'mse_decorrelated_trunc4_limit50_clf_best', 
#               'emd_decorrelated_trunc4_limit50_clf_best', 
#                'baseline_4_50',
#              'baseline_Adam_4_10',
              'baseline_Adam_4_50',
              'baseline_Adam_4_100',
#              'baseline_Adam_7_10',
#              'baseline_Adam_7_50',
#              'baseline_Adam_7_100',
#              'baseline_Nadam',
#              'baseline_RMSprop',
#              'emd',
#              'emd_clf_best',
#              'mean_squared_error',
#              'mean_squared_error_clf_best',
#              'categorical_crossentropy',
#              'categorical_crossentropy_clf_best',
#              'trunc4_limit50_clf_best',
#              'trunc4_limit50',
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
    return access(data, num) / np.clip(access(data, den), 0.0001, 999)

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
    'shallow_roc' : (lambda x : x['shallow'], makebins(0,1.2,0.0001), r'Shallow (no $p_{T}$) classifier'),
#    'baseline_Adam_4_10'  : (lambda x : x['baseline_Adam_4_10'], makebins(0,1,0.01), '(4,10)'),
#    'baseline_Adam_4_10_roc'  : (lambda x : x['baseline_Adam_4_10'], makebins(0,1,0.0001), '(4,10)'),
    'baseline_Adam_4_50'  : (lambda x : x['baseline_Adam_4_50'], makebins(0,1,0.01), '(4,50)'),
    'baseline_Adam_4_50_roc'  : (lambda x : x['baseline_Adam_4_50'], makebins(0,1,0.0001), '(4,50)'),
    'baseline_Adam_4_100'  : (lambda x : x['baseline_Adam_4_100'], makebins(0,1,0.01), '(4,100)'),
    'baseline_Adam_4_100_roc'  : (lambda x : x['baseline_Adam_4_100'], makebins(0,1,0.0001), '(4,100)'),
#    'baseline_Adam_7_10'  : (lambda x : x['baseline_Adam_7_10'], makebins(0,1,0.01), '(7,10)'),
#    'baseline_Adam_7_10_roc'  : (lambda x : x['baseline_Adam_7_10'], makebins(0,1,0.0001), '(7,10)'),
#    'baseline_Adam_7_50'  : (lambda x : x['baseline_Adam_7_50'], makebins(0,1,0.01), '(7,50)'),
#    'baseline_Adam_7_50_roc'  : (lambda x : x['baseline_Adam_7_50'], makebins(0,1,0.0001), '(7,50)'),
#    'baseline_Adam_7_100'  : (lambda x : x['baseline_Adam_7_100'], makebins(0,1,0.01), '(7,100)'),
#    'baseline_Adam_7_100_roc'  : (lambda x : x['baseline_Adam_7_100'], makebins(0,1,0.0001), '(7,100)'),
#    'trunc4_limit50_roc'  : (lambda x : x['trunc4_limit50'], makebins(0,1,0.0001), 'Decorr (4,10)'),
#    'emd'  : (lambda x : x['emd'], makebins(0,1,0.01), 'Decorr (4,10)'),
#    'emd_clf_best'  : (lambda x : x['emd_clf_best'], makebins(0,1,0.01), 'Decorr (4,10)'),
#    'emd_roc'  : (lambda x : x['emd'], makebins(0,1,0.0001), 'Decorr (4,10)'),
#    'emd_clf_best_roc'  : (lambda x : x['emd_clf_best'], makebins(0,1,0.0001), 'Decorr (4,10)'),
#    'mean_squared_error'  : (lambda x : x['mean_squared_error'], makebins(0,1,0.01), 'Decorr (4,10)'),
#    'mean_squared_error_clf_best'  : (lambda x : x['mean_squared_error_clf_best'], makebins(0,1,0.01), 'Decorr (4,10)'),
#    'mean_squared_error_roc'  : (lambda x : x['mean_squared_error'], makebins(0,1,0.0001), 'Decorr (4,10)'),
#    'mean_squared_error_clf_best_roc'  : (lambda x : x['mean_squared_error_clf_best'], makebins(0,1,0.0001), 'Decorr (4,10)'),
#    'categorical_crossentropy'  : (lambda x : x['categorical_crossentropy'], makebins(0,1,0.01), 'Decorr (4,10)'),
#    'categorical_crossentropy_clf_best'  : (lambda x : x['categorical_crossentropy_clf_best'], makebins(0,1,0.01), 'Decorr (4,10)'),
#    'categorical_crossentropy_roc'  : (lambda x : x['categorical_crossentropy'], makebins(0,1,0.0001), 'Decorr (4,10)'),
#    'categorical_crossentropy_clf_best_roc'  : (lambda x : x['categorical_crossentropy_clf_best'], makebins(0,1,0.0001), 'Decorr (4,10)'),
}

roc_vars = {
            'tau32':(r'$\tau_{32}$',0,':'),
            'tau32sd':(r'$\tau_{32}^\mathrm{SD}$',2,':'),
            'shallow_roc':('Shallow',3,':'),
            'baseline_Nadam_roc':('Baseline Nadam',12),
            'baseline_RMSprop_roc':('Baseline RMSprop',11),
            'trunc4_limit50_clf_best_roc':('Baseline 2',4,'--'),
            'trunc4_limit50_roc':('Baseline 3',4,':'),
            'emd_roc':('EMD',7),
            'emd_clf_best_roc':('EMD best',7,'--'),
            'mean_squared_error_roc':('MSE',6),
            'mean_squared_error_clf_best_roc':('MSE best',6,'--'),
            'categorical_crossentropy_roc':('CCE',5),
            'categorical_crossentropy_clf_best_roc':('CCE best',5,'--'),
            'baseline_Adam_4_10_roc':('C-LSTM (4,10)',9),
            'baseline_Adam_4_50_roc':('C-LSTM (4,50)',10),
            'baseline_Adam_4_100_roc':('C-LSTM (4,100)',11),
            'baseline_Adam_7_10_roc':('C-LSTM (7,10)',12),
            'baseline_Adam_7_50_roc':('C-LSTM (7,50)',13),
            'baseline_Adam_7_100_roc':('C-LSTM (7,100)',14),
            }

order = [
        'tau32',
        'tau32sd',
        'shallow_roc',
#        'baseline_RMSprop_roc',
#        'baseline_Nadam_roc',
#        'trunc4_limit50_clf_best_roc',
#        'trunc4_limit50_roc',
#        'emd_roc',
#        'emd_clf_best_roc',
#        'mean_squared_error_roc',
#        'mean_squared_error_clf_best_roc',
#        'categorical_crossentropy_roc',
#        'categorical_crossentropy_clf_best_roc',
#        'baseline_Adam_4_10_roc',
        'baseline_Adam_4_50_roc',
        'baseline_Adam_4_100_roc',
#        'baseline_Adam_7_10_roc',
#        'baseline_Adam_7_50_roc',
#        'baseline_Adam_7_100_roc',
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

r.clear()
r.add_vars(hists['t'],           
           hists['q'],
           roc_vars,
           order
           )
r.plot(**{'output':OUTPUT+'roc'})


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

r.clear()
r.add_vars(hists['t'],           
           hists['q'],
           roc_vars,
           order
           )
r.plot(**{'output':OUTPUT+'mass_roc'})



# get the cuts
thresholds = [0, 0.5, 0.75, 0.9, 0.99, 0.995]

def sculpting(name, f_pred):
    try:
        h = bkg_hists[name+'_roc']
    except KeyError:
        h = bkg_hists[name]
    tmp_hists = {t:{} for t in thresholds}
    f_vars2d = {
      'msd' : (lambda x : (x['singletons'][:,config.gen_singletons['msd']], f_pred(x)),
               makebins(40,400,20.),
               makebins(0,1,0.0001)),
      'pt' : (lambda x : (x['singletons'][:,config.gen_singletons['pt']], f_pred(x)),
               makebins(400,1000,50.),
               makebins(0,1,0.0001)),
      'partonm' : (lambda x : (x['singletons'][:,config.gen_singletons['partonm']], f_pred(x)),
               makebins(0,400,20.),
               makebins(0,1,0.0001)),
      }

    h2d = colls['q'].draw(components=components,
                          f_vars={}, f_vars2d=f_vars2d,
                          n_batches=n_batches, partition=partition)

    for t in thresholds:
        cut = 0
        for ib in xrange(h.bins.shape[0]):
           frac = h.integral(lo=0, hi=ib) / h.integral()
           if frac >= t:
               cut = h.bins[ib]
               break
    
        print 'For classifier=%s, threshold=%.3f reached at cut=%.3f'%(name, t, cut )
    
        for k,h2 in h2d.iteritems():
            tmp_hists[t][k] = h2.project_onto_x(min_cut=cut)

    
    colors = utils.default_colors
    for k in tmp_hists[thresholds[0]]:
        p.clear()
        p.ymin = 0.1
        p.ymax = 1e5
        for i,t in enumerate(thresholds):
            p.add_hist(tmp_hists[t][k], r'$\epsilon_\mathrm{bkg}=%.3f$'%(1-t), colors[i])
        p.plot(output=OUTPUT+'prog_'+name+'_'+k, xlabel=f_vars[k][2], logy=True)
        p.clear()
        for i,t in enumerate(thresholds):
            tmp_hists[t][k].scale()
            p.add_hist(tmp_hists[t][k], r'$\epsilon_\mathrm{bkg}=%.3f$'%(1-t), colors[i])
        p.plot(output=OUTPUT+'prognorm_'+name+'_'+k, xlabel=f_vars[k][2], logy=False)

# sculpting('emd', f_pred = f_vars['emd'][0])
# sculpting('emd_clf_best', f_pred = f_vars['emd_clf_best'][0])
# sculpting('mean_squared_error', f_pred = f_vars['mean_squared_error'][0])
# sculpting('mean_squared_error_clf_best', f_pred = f_vars['mean_squared_error_clf_best'][0])
# sculpting('categorical_crossentropy', f_pred = f_vars['categorical_crossentropy'][0])
# sculpting('categorical_crossentropy_clf_best', f_pred = f_vars['categorical_crossentropy_clf_best'][0])
# sculpting('tau32sd', f_pred = f_vars['tau32sd'][0]) 
# sculpting('baseline_Adam_7_100', f_pred = f_vars['baseline_Adam_7_100'][0])
# sculpting('shallow', f_pred = f_vars['shallow'][0])
#
