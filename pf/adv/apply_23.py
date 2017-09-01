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
# obj.DEBUG = True
# obj.truth = 'resonanceType'
# config.n_truth = 5

classifier = load_model('pretrained.h5')
shallow = load_model('tauNmsd.h5')
regularized = load_model('regularized.h5')

n_batches = 500
partition = 'test'

from keras import backend as K
K.set_image_data_format('channels_last')


def make_coll(fpath):
    coll = obj.PFSVCollection()
    coll.add_categories(['singletons', 'inclusive'], fpath) 
    return coll 

colls = {
  't' : make_coll('/home/snarayan/scratch5/baconarrays/v11_repro/PARTITION/ZprimeToTTJet_3_*_CATEGORY.npy'),
  'h' : make_coll('/data/t3serv014/bmaier/baconarrays/v1_repro//PARTITION/ZprimeToA0hToA0chichihbb_2_*_CATEGORY.npy'),
  'q' : make_coll('/home/snarayan/scratch5/baconarrays/v11_repro/PARTITION/QCD_1_*_CATEGORY.npy') 
}


# run DNN
def predict_t(data, model):
    return model.predict([data['inclusive'][:,:20,:]])[:,3]

def predict_h(data, model):
    return model.predict([data['inclusive'][:,:20,:]])[:,2]

def predict_ts(data, model):
    inputs = data['singletons'][:,[obj.singletons['tau32'],obj.singletons['tau21'],obj.singletons['msd']]]
    mus = np.array([0.5, 0.5, 75])
    sigmas = np.array([0.5, 0.5, 50])
    inputs -= mus 
    inputs /= sigmas 
    return model.predict(inputs)[:,3]

def predict_hs(data, model):
    inputs = data['singletons'][:,[obj.singletons['tau32'],obj.singletons['tau21'],obj.singletons['msd']]]
    mus = np.array([0.5, 0.5, 75])
    sigmas = np.array([0.5, 0.5, 50])
    inputs -= mus 
    inputs /= sigmas 
    return model.predict(inputs)[:,2]


f_vars = {
  'tau32' : (lambda x : x['singletons'][:,obj.singletons['tau32']], np.arange(0,1.2,0.01)),
  'tau21' : (lambda x : x['singletons'][:,obj.singletons['tau21']], np.arange(0,1.2,0.01)),
  'partonM' : (lambda x : x['singletons'][:,obj.singletons['partonM']], np.arange(0,400,5)),
  'msd'   : (lambda x : x['singletons'][:,obj.singletons['msd']], np.arange(0.,400.,20.)),
  'pt'    : (lambda x : x['singletons'][:,obj.singletons['pt']], np.arange(250.,1000.,50.)),
  'shallow_t' : (lambda x : predict_ts(x, shallow), np.arange(0,1.2,0.01)),
  'classifier_t'   : (lambda x : predict_t(x, classifier), np.arange(0,1.2,0.01)),
  'regularized_t'   : (lambda x : predict_t(x, regularized), np.arange(0,1.2,0.01)),
  'shallow_h' : (lambda x : predict_hs(x, shallow), np.arange(0,1.2,0.01)),
  'classifier_h'   : (lambda x : predict_h(x, classifier), np.arange(0,1.2,0.01)),
  'regularized_h'   : (lambda x : predict_h(x, regularized), np.arange(0,1.2,0.01)),
}

OUTPUT = '/home/snarayan/public_html/figs/badnet/th/'
system('mkdir -p '+OUTPUT)

p = utils.Plotter()
r = utils.Roccer()


# unmasked first
# hists = {}
# for k,v in colls.iteritems():
#     hists[k] = v.draw(components=['singletons', 'inclusive'],
#                       f_vars=f_vars, n_batches=n_batches, partition=partition)


# for k in hists['t']:
#     ht = hists['t'][k]
#     hq = hists['q'][k]
#     hh = hists['h'][k]
#     for h in [ht, hq, hh]:
#         h.scale()
#     p.clear()
#     p.add_hist(ht, '3-prong top', 'r')
#     p.add_hist(hh, '3-prong Higgs', 'b')
#     p.add_hist(hq, '1-prong QCD', 'k')
#     p.plot({'output':OUTPUT+'unmasked_'+k})

# r.clear()
# r.add_vars(hists['t'],
#            hists['q'],
#            {'tau32':r'$\tau_{32}$', 'classifier_t':'classifier', 
#             'regularized_t':'regularized', 'msd':r'$m_{SD}$',
#             'shallow_t':r'$\tau_{21}+\tau_{32}+m_{SD}$'},
#            )
# r.plot({'output':OUTPUT+'unmasked_top_roc'})

# r.clear()
# r.add_vars(hists['h'],
#            hists['q'],
#            {'tau32':r'$\tau_{21}$', 'classifier_h':'classifier', 
#             'regularized_h':'regularized', 'msd':r'$m_{SD}$',
#             'shallow_h':r'$\tau_{21}+\tau_{32}+m_{SD}$'},
#            )
# r.plot({'output':OUTPUT+'unmasked_higgs_roc'})


# mask the top mass
def f_mask(data):
    mass = data['singletons'][:,obj.singletons['msd']]
    return (mass > 110) & (mass < 210)


f_vars = {
  'tau32' : (lambda x : x['singletons'][:,obj.singletons['tau32']], np.arange(0,1.2,0.01)),
  'partonM' : (lambda x : x['singletons'][:,obj.singletons['partonM']], np.arange(0,400,5)),
  'msd'   : (lambda x : x['singletons'][:,obj.singletons['msd']], np.arange(0.,400.,20.)),
  'pt'    : (lambda x : x['singletons'][:,obj.singletons['pt']], np.arange(250.,1000.,50.)),
  'shallow_t' : (lambda x : predict_ts(x, shallow), np.arange(0,1.2,0.01)),
  'classifier_t'   : (lambda x : predict_t(x, classifier), np.arange(0,1.2,0.01)),
  'regularized_t'   : (lambda x : predict_t(x, regularized), np.arange(0,1.2,0.01)),
 }

# hists = {}
# for k,v in colls.iteritems():
#     hists[k] = v.draw(components=['singletons', 'inclusive'],
#                       f_vars=f_vars, n_batches=n_batches, partition=partition, f_mask=f_mask)


# for k in hists['t']:
#     ht = hists['t'][k]
#     hq = hists['q'][k]
#     hh = hists['h'][k]
#     for h in [ht, hq, hh]:
#         h.scale()
#     p.clear()
#     p.add_hist(ht, '3-prong top', 'r')
#     p.add_hist(hh, '3-prong Higgs', 'b')
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



exit(0)

# get the cuts
threshold = 0.98
h = hists_qcd['classifier']
classifier_cut = 0
for ib in xrange(h.bins.shape[0]):
    frac = h.integral(lo=0, hi=ib) / h.integral()
    if frac >= threshold:
        classifier_cut = h.bins[ib]
        break

h = hists_qcd['regularized']
regularized_cut = 0
for ib in xrange(h.bins.shape[0]):
    frac = h.integral(lo=0, hi=ib) / h.integral()
    if frac >= threshold:
        regularized_cut = h.bins[ib]
        break

h = hists_qcd['tau32']
tau32_cut = 0
for ib in xrange(h.bins.shape[0]):
    frac = h.integral(lo=(h.bins.shape[0]-ib), hi=None) / h.integral()
    if frac >= threshold:
        tau32_cut = h.bins[h.bins.shape[0]-ib]
        break

print 'classifier', classifier_cut
print 'regularized', regularized_cut
print 'tau32', tau32_cut

# mask pretrain
def f_mask(data, model, cut):
    return predict(data, model) > cut

hists_top = top_4.draw(components=['singletons', 'inclusive'],
                       f_vars=f_vars, n_batches=n_batches, partition=partition,
                       f_mask = lambda x : f_mask(x, classifier, classifier_cut))
hists_qcd = qcd_0.draw(components=['singletons', 'inclusive'],
                       f_vars=f_vars, n_batches=n_batches, partition=partition, 
                       f_mask = lambda x : f_mask(x, classifier, classifier_cut))

for k in hists_top:
    htop = hists_top[k]
    hqcd = hists_qcd[k]
    htop.scale() 
    hqcd.scale()
    p.clear()
    p.add_hist(htop, '3-prong', 'r')
    p.add_hist(hqcd, '1-prong', 'k')
    p.plot({'output':OUTPUT+'classifier_'+k})


hists_top = top_4.draw(components=['singletons', 'inclusive'],
                       f_vars=f_vars, n_batches=n_batches, partition=partition,
                       f_mask = lambda x : f_mask(x, regularized, regularized_cut))
hists_qcd = qcd_0.draw(components=['singletons', 'inclusive'],
                       f_vars=f_vars, n_batches=n_batches, partition=partition,
                       f_mask = lambda x : f_mask(x, regularized, regularized_cut))

for k in hists_top:
    htop = hists_top[k]
    hqcd = hists_qcd[k]
    htop.scale() 
    hqcd.scale()
    p.clear()
    p.add_hist(htop, '3-prong', 'r')
    p.add_hist(hqcd, '1-prong', 'k')
    p.plot({'output':OUTPUT+'regularized_'+k})


# mask tau32
def f_mask(data, cut):
    return data['singletons'][:,obj.singletons['tau32']] < cut 

hists_top = top_4.draw(components=['singletons', 'inclusive'],
                       f_vars=f_vars, n_batches=n_batches, 
                       f_mask = lambda x : f_mask(x, tau32_cut))
hists_qcd = qcd_0.draw(components=['singletons', 'inclusive'],
                       f_vars=f_vars, n_batches=n_batches, 
                       f_mask = lambda x : f_mask(x, tau32_cut))

for k in hists_top:
    htop = hists_top[k]
    hqcd = hists_qcd[k]
    htop.scale() 
    hqcd.scale()
    p.clear()
    p.add_hist(htop, '3-prong', 'r')
    p.add_hist(hqcd, '1-prong', 'k')
    p.plot({'output':OUTPUT+'tau32_'+k})
