#!/usr/local/bin/python2.7

from sys import exit 
from os import environ, system
environ['KERAS_BACKEND'] = 'tensorflow'
environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
environ["CUDA_VISIBLE_DEVICES"] = ""


import numpy as np
import utils
import adversarial


import obj 
import config 
# config.DEBUG = True
config.n_truth = 5
config.truth = 'resonanceType'

n_batches = 1000
partition = 'test'

OUTPUT = '/home/snarayan/public_html/figs/badnet/akt_ordering/'
system('mkdir -p '+OUTPUT)

p = utils.Plotter()
r = utils.Roccer()

#components=['singletons', 'inclusive', 'nn1', 'nn2']
components=['singletons', 'inclusive', 'nn']


def make_coll(fpath):
    coll = obj.PFSVCollection()
    coll.add_categories(components, fpath) 
    return coll 

colls = {
  't' : make_coll('/fastscratch/snarayan/baconarrays/v12_repro/PARTITION/ZprimeToTTJet_4_*_CATEGORY.npy'),
  'q' : make_coll('/fastscratch/snarayan/baconarrays/v12_repro/PARTITION/QCD_0_*_CATEGORY.npy') 
}


# run DNN
def predict(data, model):
    return data['nn'][:,model]
   
def predict1(data, model):
    return data['nn1'][:,model]
    
def predict2(data):
    return data['nn2']

f_vars = {
  'tau32' : (lambda x : x['singletons'][:,obj.singletons['tau32']], np.arange(0,1.2,0.01)),
  'tau21' : (lambda x : x['singletons'][:,obj.singletons['tau21']], np.arange(0,1.2,0.01)),
  'partonM' : (lambda x : x['singletons'][:,obj.singletons['partonM']], np.arange(0,400,5)),
  'msd'   : (lambda x : x['singletons'][:,obj.singletons['msd']], np.arange(0.,400.,10.)),
  'pt'    : (lambda x : x['singletons'][:,obj.singletons['pt']], np.arange(250.,1000.,50.)),
  'shallow_t' : (lambda x : predict(x, 0), np.arange(0,1.2,0.001)),
  'classifier_t'   : (lambda x : predict(x, 1), np.arange(0,1.2,0.001)),
  'regularized_t'   : (lambda x : predict(x, 2), np.arange(0,1.2,0.001)),
}


# unmasked first
hists = {}
for k,v in colls.iteritems():
    hists[k] = v.draw(components=components,
                      f_vars=f_vars, n_batches=n_batches, partition=partition)


for k in hists['t']:
    ht = hists['t'][k]
    hq = hists['q'][k]
    # hh = hists['h'][k]
    # for h in [ht, hq, hh]:
    for h in [ht, hq]:
        h.scale()
    p.clear()
    p.add_hist(ht, '3-prong top', 'r')
    # p.add_hist(hh, '2-prong Higgs', 'b')
    p.add_hist(hq, '1-prong QCD', 'k')
    p.plot({'output':OUTPUT+'unmasked_'+k})

r.clear()
r.add_vars(hists['t'],
           hists['q'],
           {'tau32':r'$\tau_{32}$', 'classifier_t':'classifier', 
            'regularized_t':'regularized', 'msd':r'$m_{SD}$',
            'shallow_t':r'$\tau_{21}+\tau_{32}+m_{SD}$'},
           )
r.plot({'output':OUTPUT+'unmasked_top_roc'})

# r.clear()
# r.add_vars(hists['h'],
#            hists['q'],
#            {'tau32':r'$\tau_{21}$', 'classifier_h':'classifier', 
#             'regularized_h':'regularized', 'msd':r'$m_{SD}$',
#             'shallow_h':r'$\tau_{21}+\tau_{32}+m_{SD}$'},
#            )
# r.plot({'output':OUTPUT+'unmasked_higgs_roc'})


n_batches = 400

# # mask the top mass
# def f_mask(data):
#     mass = data['singletons'][:,obj.singletons['msd']]
#     return (mass > 110) & (mass < 210)


# f_vars = {
#   'tau32' : (lambda x : x['singletons'][:,obj.singletons['tau32']], np.arange(0,1.2,0.01)),
#   'partonM' : (lambda x : x['singletons'][:,obj.singletons['partonM']], np.arange(0,400,5)),
#   'msd'   : (lambda x : x['singletons'][:,obj.singletons['msd']], np.arange(0.,400.,10.)),
#   'pt'    : (lambda x : x['singletons'][:,obj.singletons['pt']], np.arange(250.,1000.,50.)),
#   'shallow_t' : (lambda x : predict(x, 0), np.arange(0,1.2,0.001)),
#   'classifier_t'   : (lambda x : predict(x, 1), np.arange(0,1.2,0.001)),
#   'regularized_t'   : (lambda x : predict(x, 2), np.arange(0,1.2,0.001)),
#  }

# hists = {}
# for k,v in colls.iteritems():
#     hists[k] = v.draw(components=['singletons', 'inclusive', 'nn'],
#                       f_vars=f_vars, n_batches=n_batches, partition=partition, f_mask=f_mask)


# for k in hists['t']:
#     ht = hists['t'][k]
#     hq = hists['q'][k]
#     # hh = hists['h'][k]
#     for h in [ht, hq]:
#         h.scale()
#     p.clear()
#     p.add_hist(ht, '3-prong top', 'r')
#     # p.add_hist(hh, '3-prong Higgs', 'b')
#     p.add_hist(hq, '1-prong QCD', 'k')
#     p.plot({'output':OUTPUT+'topmass_'+k})

# r.clear()
# r.add_vars(hists['t'],
#            hists['q'],
#            {'tau32':r'$\tau_{32}$', 'classifier_t':'classifier', 
#             'regularized_t':'regularized', 'msd':r'$m_{SD}$',
#             'shallow_t':r'$\tau_{21}+\tau_{32}+m_{SD}$'},
#            )
# r.plot({'output':OUTPUT+'topmass_top_roc'})


'''
# mask the higgs mass
def f_mask(data):
    mass = data['singletons'][:,obj.singletons['msd']]
    return (mass > 90) & (mass < 140)


f_vars = {
  'tau21' : (lambda x : x['singletons'][:,obj.singletons['tau21']], np.arange(0,1.2,0.01)),
  'partonM' : (lambda x : x['singletons'][:,obj.singletons['partonM']], np.arange(0,400,5)),
  'msd'   : (lambda x : x['singletons'][:,obj.singletons['msd']], np.arange(0.,400.,20.)),
  'pt'    : (lambda x : x['singletons'][:,obj.singletons['pt']], np.arange(250.,1000.,50.)),
  'shallow_h' : (lambda x : predict_hs(x, shallow), np.arange(0,1.2,0.01)),
  'classifier_h'   : (lambda x : predict_h(x, classifier), np.arange(0,1.2,0.01)),
  'regularized_h'   : (lambda x : predict_h(x, regularized), np.arange(0,1.2,0.01)),
 }

hists = {}
for k,v in colls.iteritems():
    hists[k] = v.draw(components=['singletons', 'inclusive'],
                      f_vars=f_vars, n_batches=n_batches, partition=partition, f_mask=f_mask)


for k in hists['t']:
    ht = hists['t'][k]
    hq = hists['q'][k]
    hh = hists['h'][k]
    for h in [ht, hq, hh]:
        h.scale()
    p.clear()
    p.add_hist(ht, '3-prong top', 'r')
    p.add_hist(hh, '3-prong Higgs', 'b')
    p.add_hist(hq, '1-prong QCD', 'k')
    p.plot({'output':OUTPUT+'higgsmass_'+k})

r.clear()
r.add_vars(hists['h'],
           hists['q'],
           {'tau21':r'$\tau_{21}$', 'classifier_h':'classifier', 
            'regularized_h':'regularized', 'msd':r'$m_{SD}$',
            'shallow_h':r'$\tau_{21}+\tau_{32}+m_{SD}$'},
           )
r.plot({'output':OUTPUT+'higgsmass_higgs_roc'})

'''

# get the cuts
threshold = 0.95
h = hists['q']['classifier_t']
classifier_cut = 0
for ib in xrange(h.bins.shape[0]):
    frac = h.integral(lo=0, hi=ib) / h.integral()
    if frac >= threshold:
        classifier_cut = h.bins[ib]
        break

h = hists['q']['regularized_t']
regularized_cut = 0
for ib in xrange(h.bins.shape[0]):
    frac = h.integral(lo=0, hi=ib) / h.integral()
    if frac >= threshold:
        regularized_cut = h.bins[ib]
        break

h = hists['q']['shallow_t']
shallow_cut = 0
for ib in xrange(h.bins.shape[0]):
    frac = h.integral(lo=0, hi=ib) / h.integral()
    if frac >= threshold:
        shallow_cut = h.bins[ib]
        break

print 'classifier', classifier_cut
print 'regularized', regularized_cut
print 'shallow', shallow_cut

# mask pretrain
def f_mask(data, model, cut):
    return predict(data, model) > cut
# def f_mask(data, model, cut):
#     return predict2(data) > cut

n_batches = 10000

# hists['t'] = colls['t'].draw(components=['singletons', 'inclusive'],
#                        f_vars=f_vars, n_batches=n_batches, partition=partition,
#                        f_mask = lambda x : f_mask(x, classifier, classifier_cut))
hists['q'] = colls['q'].draw(components=components,
                       f_vars=f_vars, n_batches=n_batches, partition=partition, 
                       f_mask = lambda x : f_mask(x, 1, classifier_cut))

for k in hists['q']:
#    htop = hists['t'][k]
    hqcd = hists['q'][k]
#    htop.scale() 
    hqcd.scale()
    p.clear()
#    p.add_hist(htop, '3-prong', 'r')
    p.add_hist(hqcd, '1-prong', 'k')
    p.plot({'output':OUTPUT+'classifier_'+k})


# def f_mask(data, model, cut):
#     return predict1(data, model) > cut

#hists['t'] = colls['t'].draw(components=['singletons', 'inclusive'],
#                       f_vars=f_vars, n_batches=n_batches, partition=partition,
#                       f_mask = lambda x : f_mask(x, regularized, regularized_cut))
hists['q'] = colls['q'].draw(components=components,
                       f_vars=f_vars, n_batches=n_batches, partition=partition,
                       f_mask = lambda x : f_mask(x, 2, regularized_cut))

for k in hists['q']:
#    htop = hists['t'][k]
    hqcd = hists['q'][k]
#    htop.scale() 
    hqcd.scale()
    p.clear()
#    p.add_hist(htop, '3-prong', 'r')
    p.add_hist(hqcd, '1-prong', 'k')
    p.plot({'output':OUTPUT+'regularized_'+k})


# mask tau32

#hists['t'] = colls['t'].draw(components=['singletons', 'inclusive'],
#                       f_vars=f_vars, n_batches=n_batches, 
#                       f_mask = lambda x : f_mask(x, shallow, shallow_cut))
hists['q'] = colls['q'].draw(components=components,
                       f_vars=f_vars, n_batches=n_batches, 
                       f_mask = lambda x : f_mask(x, 0, shallow_cut))

for k in hists['q']:
#    htop = hists['t'][k]
    hqcd = hists['q'][k]
#    htop.scale() 
    hqcd.scale()
    p.clear()
#    p.add_hist(htop, '3-prong', 'r')
    p.add_hist(hqcd, '1-prong', 'k')
    p.plot({'output':OUTPUT+'shallow_'+k})

# now mask the higgs classifiers
'''
threshold = 0.99
h = hists['q']['classifier_h']
classifier_cut = 0
for ib in xrange(h.bins.shape[0]):
    frac = h.integral(lo=0, hi=ib) / h.integral()
    if frac >= threshold:
        classifier_cut = h.bins[ib]
        break

h = hists['q']['regularized_h']
regularized_cut = 0
for ib in xrange(h.bins.shape[0]):
    frac = h.integral(lo=0, hi=ib) / h.integral()
    if frac >= threshold:
        regularized_cut = h.bins[ib]
        break

h = hists['q']['shallow_h']
shallow_cut = 0

for ib in xrange(h.bins.shape[0]):
    frac = h.integral(lo=0, hi=ib) / h.integral()
    if frac >= threshold:
        shallow_cut =  h.bins[ib]
        break

print 'classifier', classifier_cut
print 'regularized', regularized_cut
print 'shallow', shallow_cut

# mask pretrain
def f_mask(data, model, cut):
    return predict_h(data, model) > cut

hists['h'] = colls['h'].draw(components=['singletons', 'inclusive'],
                       f_vars=f_vars, n_batches=n_batches, partition=partition,
                       f_mask = lambda x : f_mask(x, classifier, classifier_cut))
hists['q'] = colls['q'].draw(components=['singletons', 'inclusive'],
                       f_vars=f_vars, n_batches=n_batches, partition=partition, 
                       f_mask = lambda x : f_mask(x, classifier, classifier_cut))

for k in hists['h']:
    htop = hists['h'][k]
    hqcd = hists['q'][k]
    htop.scale() 
    hqcd.scale()
    p.clear()
    p.add_hist(htop, '2-prong', 'r')
    p.add_hist(hqcd, '1-prong', 'k')
    p.plot({'output':OUTPUT+'classifierh_'+k})


hists['h'] = colls['h'].draw(components=['singletons', 'inclusive'],
                       f_vars=f_vars, n_batches=n_batches, partition=partition,
                       f_mask = lambda x : f_mask(x, regularized, regularized_cut))
hists['q'] = colls['q'].draw(components=['singletons', 'inclusive'],
                       f_vars=f_vars, n_batches=n_batches, partition=partition,
                       f_mask = lambda x : f_mask(x, regularized, regularized_cut))

for k in hists['h']:
    htop = hists['h'][k]
    hqcd = hists['q'][k]
    htop.scale() 
    hqcd.scale()
    p.clear()
    p.add_hist(htop, '2-prong', 'r')
    p.add_hist(hqcd, '1-prong', 'k')
    p.plot({'output':OUTPUT+'regularizedh_'+k})


# mask tau32
def f_mask(data, model, cut):
    return predict_hs(data, model) > cut

hists['h'] = colls['h'].draw(components=['singletons', 'inclusive'],
                       f_vars=f_vars, n_batches=n_batches, 
                       f_mask = lambda x : f_mask(x, shallow, shallow_cut))
hists['q'] = colls['q'].draw(components=['singletons', 'inclusive'],
                       f_vars=f_vars, n_batches=n_batches, 
                       f_mask = lambda x : f_mask(x, shallow, shallow_cut))

for k in hists['h']:
    htop = hists['h'][k]
    hqcd = hists['q'][k]
    htop.scale() 
    hqcd.scale()
    p.clear()
    p.add_hist(htop, '2-prong', 'r')
    p.add_hist(hqcd, '1-prong', 'k')
    p.plot({'output':OUTPUT+'shallowh_'+k})

'''
