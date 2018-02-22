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
              'baseline_trunc4_limit50_best', 
              'decorrelated_trunc4_limit50_best', 
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
    'tau32sd' : (lambda x : div(x, 'tau3sd', 'tau2sd'), np.arange(0,1.2,0.01), r'$\tau_{32}^\mathrm{sd}$'),
    'partonm' : (lambda x : access(x, 'partonm'), np.arange(0,400,5), 'Parton mass [GeV]'),
    'msd'     : (lambda x : access(x, 'msd'), np.arange(0.,400.,20.), r'$m_\mathrm{SD}$ [GeV]'),
    'pt'        : (lambda x : access(x, 'pt'), np.arange(250.,1000.,50.), r'$p_\mathrm{T}$ [GeV]'),
    'shallow_nopt' : (lambda x : x['shallow_nopt'], np.arange(0,1.2,0.01), r'Shallow (no $p_{T}$) classifier'),
    'shallow_nopt_roc' : (lambda x : x['shallow_nopt'], np.arange(0,1.2,0.0001), r'Shallow (no $p_{T}$) classifier'),
    'baseline_4_50'  : (lambda x : x['baseline_trunc4_limit50_best'], np.arange(0,1,0.01), 'Baseline (4,10)'),
    'decorrelated_4_50'  : (lambda x : x['decorrelated_trunc4_limit50_best'], np.arange(0,1,0.01), 'Decorr (4,10)'),
    'baseline_4_50_roc'  : (lambda x : x['baseline_trunc4_limit50_best'], np.arange(-0.2,1.2,0.00001), 'Baseline (4,10)'),
    'decorrelated_4_50_roc'  : (lambda x : x['decorrelated_trunc4_limit50_best'], np.arange(-0.2,1.2,0.00001), 'Decorr (4,10)'),
}

roc_vars = {
            'tau32':(r'$\tau_{32}$',0,':'),
            'tau32sd':(r'$\tau_{32}^\mathrm{SD}$',2,':'),
            'baseline_4_50_roc':('Baseline (4,50)',4),
            'decorrelated_4_50_roc':('Decorr (4,50)',4,'--'),
            'lstm4_50_roc':('LSTM (4,50)',5),
            'lstm4_500_roc':('LSTM (4,100)',6),
            'lstm7_50_roc':('LSTM (7,50)',7),
            'lstm7_100_roc':('LSTM (7,100)',3),
            'shallow_nopt_roc':('Shallow',9,'--'),
            'dense7_50_roc':('Dense (7,50)',10,'--'),
            'dense4_500_roc':('Dense (4,100)',11,'--'),
            }

order = [
        'tau32',
        'tau32sd',
        'shallow_nopt_roc',
        'baseline_4_50_roc',
        'decorrelated_4_50_roc',
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




# get the cuts
thresholds = [0, 0.5, 0.75, 0.9, 0.99, 0.999]

def sculpting(name, f_pred):
    try:
        h = hists['q'][name+'_roc']
    except KeyError:
        h = hists['q'][name]
    tmp_hists = {t:{} for t in thresholds}
    f_vars2d = {
      'msd' : (lambda x : (x['singletons'][:,config.gen_singletons['msd']], f_pred(x)),
               np.arange(40,400,20.),
               np.arange(0,1,0.0001)),
      'pt' : (lambda x : (x['singletons'][:,config.gen_singletons['pt']], f_pred(x)),
               np.arange(400,1000,50.),
               np.arange(0,1,0.0001)),
      'partonm' : (lambda x : (x['singletons'][:,config.gen_singletons['partonm']], f_pred(x)),
               np.arange(0,400,20.),
               np.arange(0,1,0.0001)),
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

sculpting('tau32sd', f_pred = f_vars['tau32sd'][0]) 
sculpting('decorrelated_4_50', f_pred = f_vars['decorrelated_4_50'][0])
sculpting('baseline_4_50', f_pred = f_vars['baseline_4_50'][0])

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


