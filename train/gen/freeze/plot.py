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

components = [
              'singletons',
              'shallow', 
              'baseline2_7_100',
              'kltest_7_100',
              'categorical_crossentropy2_7_100',
              'categorical_crossentropytest2_7_100',
              'categorical_crossentropytesttest2_7_100',
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
    'msd'     : (lambda x : access(x, 'msd'), makebins(40.,400.,20.), r'$m_\mathrm{SD}$ [GeV]'),
    'pt'        : (lambda x : access(x, 'pt'), makebins(400.,1000.,50.), r'$p_\mathrm{T}$ [GeV]'),
    'shallow' : (lambda x : x['shallow'], makebins(0,1.2,0.01), r'Shallow (no $p_{T}$) classifier'),
    'shallow_roc' : (lambda x : x['shallow'], makebins(0,1.2,0.00003), r'Shallow (no $p_{T}$) classifier'),
    'baseline2_7_100'  : (lambda x : x['baseline2_7_100'], makebins(0,1.01,0.01), '(7,50)'),
    'baseline2_7_100_roc'  : (lambda x : x['baseline2_7_100'], makebins(0,1.01,0.00003), '(7,50)'),
    'categorical_crossentropy2_7_100'  : (lambda x : x['categorical_crossentropy2_7_100'], makebins(0,1.01,0.01), '(7,50)'),
    'categorical_crossentropy2_7_100_roc'  : (lambda x : x['categorical_crossentropy2_7_100'], makebins(0,1.01,0.00003), '(7,50)'),
    'kltest_7_100'  : (lambda x : x['kltest_7_100'], makebins(0,1.01,0.01), '(7,50)'),
    'kltest_7_100_roc'  : (lambda x : x['kltest_7_100'], makebins(0,1.01,0.00003), '(7,50)'),
    'categorical_crossentropytest2_7_100'  : (lambda x : x['categorical_crossentropytest2_7_100'], makebins(0,1.01,0.01), '(7,50)'),
    'categorical_crossentropytest2_7_100_roc'  : (lambda x : x['categorical_crossentropytest2_7_100'], makebins(0,1.01,0.00003), '(7,50)'),
    'categorical_crossentropytesttest2_7_100'  : (lambda x : x['categorical_crossentropytesttest2_7_100'], makebins(0,1.01,0.01), '(7,50)'),
    'categorical_crossentropytesttest2_7_100_roc'  : (lambda x : x['categorical_crossentropytesttest2_7_100'], makebins(0,1.01,0.00003), '(7,50)'),
}

roc_vars = {
            'tau32':(r'$\tau_{32}$',0,':'),
            'tau32sd':(r'$\tau_{32}^\mathrm{SD}$',2,':'),
            'shallow_roc':('Shallow',3,':'),
 #           'categorical_crossentropytesttest2_7_100_roc':('C-LSTM (7,50)',14),
            'categorical_crossentropytest2_7_100_roc':('Weak decorr.',12,'-'),
            'categorical_crossentropy2_7_100_roc':('Strong decorr.',11,'-'),
#            'kltest_7_100_roc':('kltest_7_100',10),
            'baseline2_7_100_roc':('C-LSTM (7,50)',3),
            }

order = [
        'tau32',
        'tau32sd',
        'shallow_roc',
#        'categorical_crossentropytesttest2_7_100_roc',
#        'kltest_7_100_roc',
        'categorical_crossentropy2_7_100_roc',
        'categorical_crossentropytest2_7_100_roc',
        'baseline2_7_100_roc',
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

r1.clear()
r1.add_vars(hists['t'],           
           hists['q'],
           roc_vars,
           order
           )
r1.plot(**{'output':OUTPUT+'roc'})

h_top_mass = hists['t']['msd'].clone()
h_top_mass.scale(5)

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

r2.clear()
r2.add_vars(hists['t'],           
           hists['q'],
           roc_vars,
           order
           )
r2.plot(**{'output':OUTPUT+'mass_roc'})

bkg_hists = {k:v for k,v in hists['q'].iteritems()}


# get the cuts
thresholds = [0, 0.5, 0.9, 0.95, 0.99]

def sculpting(name, f_pred):
    try:
        h = bkg_hists[name+'_roc']
    except KeyError:
        h = bkg_hists[name]
    tmp_hists = {t:{} for t in thresholds}

#    f_vars2d = {
#      'msd' : (lambda x : (x['singletons'][:,config.gen_singletons['msd']], f_pred(x)),
#               makebins(-10,400,20.),
#               makebins(0,1.2,0.0001)),
#      'pt' : (lambda x : (x['singletons'][:,config.gen_singletons['pt']], f_pred(x)),
#               makebins(400,1200,50.),
#               makebins(0,1.2,0.0001)),
#      'partonm' : (lambda x : (x['singletons'][:,config.gen_singletons['partonm']], f_pred(x)),
#               makebins(-10,400,20.),
#               makebins(0,1.2,0.0001)),
#      }
#
#    h2d = colls['q'].draw(components=components,
#                          f_vars={}, f_vars2d=f_vars2d,
#                          n_batches=n_batches, partition=partition)

    oned_vars = {k : f_vars[k] for k in ['msd', 'partonm', 'pt']}

    for t in thresholds:
        cut = h.quantile(t, interp=True)
    
        print 'For classifier=%s, threshold=%.3f reached at cut=%.3f'%(name, t, cut )

        def _my_mask(x):
            return f_pred(x) > cut

        tmp_hists[t] = colls['q'].draw(components=components,
                                       f_vars=oned_vars,
                                       n_batches=n_batches,
                                       partition=partition,
                                       f_mask=_my_mask)

#        for k,v in tmp_hists[t].iteritems():
#            print name, t, k, v.integral()

#         for k,h2 in h2d.iteritems():
#             tmp_hists[t][k] = h2.project_onto_x(min_cut=cut)
#             print t, k, tmp_hists[t][k].integral(), '/', h.integral()

    
    colors = utils.default_colors
    for k in tmp_hists[thresholds[0]]:
        p.clear()
        p.ymin = 0.05
        p.ymax = 5e5
        for i,t in enumerate(thresholds):
            p.add_hist(tmp_hists[t][k], r'$\epsilon_\mathrm{bkg}=%.2f$'%(1-t), colors[i])
        p.add_hist(h_top_mass, 'Top mass shape', 'k')
        p.plot(output=OUTPUT+'prog_'+name+'_'+k, xlabel=f_vars[k][2], logy=True)
        p.clear()
        for i,t in enumerate(thresholds):
            tmp_hists[t][k].scale()
            p.add_hist(tmp_hists[t][k], r'$\epsilon_\mathrm{bkg}=%.2f$'%(1-t), colors[i])
        p.plot(output=OUTPUT+'prognorm_'+name+'_'+k, xlabel=f_vars[k][2], logy=False)


sculpting('kltest_7_100', f_pred = f_vars['kltest_7_100'][0])
sculpting('categorical_crossentropytesttest2_7_100', f_pred = f_vars['categorical_crossentropytesttest2_7_100'][0])
sculpting('categorical_crossentropytest2_7_100', f_pred = f_vars['categorical_crossentropytest2_7_100'][0])
sculpting('baseline2_7_100', f_pred = f_vars['baseline2_7_100'][0])
sculpting('categorical_crossentropy2_7_100', f_pred = f_vars['categorical_crossentropy2_7_100'][0])
sculpting('tau32sd', f_pred = f_vars['tau32sd'][0]) 
sculpting('shallow', f_pred = f_vars['shallow'][0])
#
