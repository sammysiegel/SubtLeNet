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

n_batches = 300
partition = 'test'

OUTPUT = '/home/snarayan/public_html/figs/badnet/akt_ordering_conv/'
system('mkdir -p '+OUTPUT)

p = utils.Plotter()
r = utils.Roccer()

#components=['singletons', 'inclusive', 'nn1', 'nn2']
components=['singletons', 'inclusive', 'nn', 'nn_conv']


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

def predict_conv(data, model):
    return data['nn_conv'][:,model]
   
def predict1(data, model):
    return data['nn1'][:,model]
    
def predict2(data):
    return data['nn2']

f_vars = {
  'tau32' : (lambda x : x['singletons'][:,obj.singletons['tau32']], np.arange(0,1.2,0.01)),
  'tau21' : (lambda x : x['singletons'][:,obj.singletons['tau21']], np.arange(0,1.2,0.01)),
  'partonM' : (lambda x : x['singletons'][:,obj.singletons['partonM']], np.arange(0,400,5)),
  'msd'   : (lambda x : x['singletons'][:,obj.singletons['msd']], np.arange(0.,400.,20.)),
  'pt'    : (lambda x : x['singletons'][:,obj.singletons['pt']], np.arange(250.,1000.,50.)),
  'shallow_t' : (lambda x : predict(x, 0), np.arange(0,1.2,0.001)),
  'classifier_t'   : (lambda x : predict(x, 1), np.arange(0,1.2,0.001)),
  'regularized_t'   : (lambda x : predict(x, 2), np.arange(0,1.2,0.001)),
  'classifier_conv_t'   : (lambda x : predict_conv(x, 0), np.arange(0,1.2,0.001)),
  'regularized_conv_t'   : (lambda x : predict_conv(x, 1), np.arange(0,1.2,0.001)),
}

f_vars2d = {
  'correlation_reg' : (lambda x : (x['singletons'][:,obj.singletons['msd']], predict(x, 2)),
                       np.arange(0,400,10),
                       np.arange(0,1.2,0.001)),
  'correlation_class' : (lambda x : (x['singletons'][:,obj.singletons['msd']], predict(x, 1)),
                       np.arange(0,400,10),
                       np.arange(0,1.2,0.001)),
}

# unmasked first
hists = {}
hists2d = {}
for k,v in colls.iteritems():
    hists[k],hists2d[k] = v.draw(components=components,
                                 f_vars=f_vars, f_vars2d=f_vars2d,
                                 n_batches=n_batches, partition=partition)


hists2d['q']['correlation_reg'].plot(xlabel=r'$m_{SD}$', ylabel='Regularized NN', 
                                     output=OUTPUT+'correlation_reg')
hists2d['q']['correlation_class'].plot(xlabel=r'$m_{SD}$', ylabel='Classifier NN', 
                                       output=OUTPUT+'correlation_class')

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
    p.plot(**{'output':OUTPUT+'unmasked_'+k})

r.clear()
r.add_vars(hists['t'],
           hists['q'],
           {'tau32':r'$\tau_{32}$', 'classifier_t':'classifier', 
            'regularized_t':'regularized', 'msd':r'$m_{SD}$',
            'classifier_conv_t':'classifier conv',
            'regularized_conv_t':'regularized conv',
            'shallow_t':r'$\tau_{21}+\tau_{32}+m_{SD}$'},
           )
r.plot(**{'output':OUTPUT+'unmasked_top_roc'})

# mask the top mass
def f_mask(data):
    mass = data['singletons'][:,obj.singletons['msd']]
    return (mass > 110) & (mass < 210)


f_vars = {
  'tau32' : (lambda x : x['singletons'][:,obj.singletons['tau32']], np.arange(0,1.2,0.01)),
  'partonM' : (lambda x : x['singletons'][:,obj.singletons['partonM']], np.arange(0,400,5)),
  'msd'   : (lambda x : x['singletons'][:,obj.singletons['msd']], np.arange(0.,400.,20.)),
  'pt'    : (lambda x : x['singletons'][:,obj.singletons['pt']], np.arange(250.,1000.,50.)),
  'shallow_t' : (lambda x : predict(x, 0), np.arange(0,1.2,0.001)),
  'classifier_t'   : (lambda x : predict(x, 1), np.arange(0,1.2,0.001)),
  'regularized_t'   : (lambda x : predict(x, 2), np.arange(0,1.2,0.001)),
  'classifier_conv_t'   : (lambda x : predict_conv(x, 0), np.arange(0,1.2,0.001)),
  'regularized_conv_t'   : (lambda x : predict_conv(x, 1), np.arange(0,1.2,0.001)),
 }

hists = {}
for k,v in colls.iteritems():
    hists[k] = v.draw(components=components,
                      f_vars=f_vars, n_batches=n_batches, partition=partition, f_mask=f_mask)


for k in hists['t']:
    ht = hists['t'][k]
    hq = hists['q'][k]
    # hh = hists['h'][k]
    for h in [ht, hq]:
        h.scale()
    p.clear()
    p.add_hist(ht, '3-prong top', 'r')
    # p.add_hist(hh, '3-prong Higgs', 'b')
    p.add_hist(hq, '1-prong QCD', 'k')
    p.plot(**{'output':OUTPUT+'topmass_'+k})

r.clear()
r.add_vars(hists['t'],
           hists['q'],
           {'tau32':r'$\tau_{32}$', 'classifier_t':'classifier', 
            'regularized_t':'regularized', 'msd':r'$m_{SD}$',
            'classifier_conv_t':'classifier conv',
            'regularized_conv_t':'regularized conv',
            'shallow_t':r'$\tau_{21}+\tau_{32}+m_{SD}$'},
           )
r.plot(**{'output':OUTPUT+'topmass_top_roc'})


# exit(0)

# get the cuts
threshold = 0.98

def sculpting(name, f_mask):
  h = hists['q'][name]
  cut = 0
  for ib in xrange(h.bins.shape[0]):
      frac = h.integral(lo=0, hi=ib) / h.integral()
      if frac >= threshold:
          cut = h.bins[ib]
          break

  print name, cut 

  f_vars = {
  'partonM' : (lambda x : x['singletons'][:,obj.singletons['partonM']], np.arange(0,400,5)),
  'msd'   : (lambda x : x['singletons'][:,obj.singletons['msd']], np.arange(0.,400.,20.)),
  'pt'    : (lambda x : x['singletons'][:,obj.singletons['pt']], np.arange(250.,1000.,50.)),
  }

  tmp_hists = {}
  tmp_hists['q'] = colls['q'].draw(components=components,
                                   f_vars=f_vars, n_batches=n_batches, partition=partition, 
                                   f_mask = lambda x : f_mask(x, cut))
  tmp_hists['t'] = colls['t'].draw(components=components,
                                   f_vars=f_vars, n_batches=n_batches, partition=partition, 
                                   f_mask = lambda x : f_mask(x, cut))

  for k in tmp_hists['q']:
      htop = tmp_hists['t'][k]
      hqcd = tmp_hists['q'][k]
      htop.scale() 
      hqcd.scale()
      p.clear()
      p.add_hist(htop, '3-prong', 'r')
      p.add_hist(hqcd, '1-prong', 'k')
      p.plot(**{'output':OUTPUT+name+'_'+k})

def f_mask_base(data, model, cut):
    return predict(data, model) > cut
def f_mask_conv_base(data, model, cut):
    return predict_conv(data, model) > cut

sculpting('regularized_conv_t', f_mask = lambda d, c : f_mask_conv_base(d, 1, c))
sculpting('classifier_t', f_mask = lambda d, c : f_mask_base(d, 1, c))
sculpting('classifier_conv_t', f_mask = lambda d, c : f_mask_conv_base(d, 0, c))
sculpting('regularized_t', f_mask = lambda d, c : f_mask_base(d, 2, c))
sculpting('shallow_t', f_mask = lambda d, c : f_mask_base(d, 0, c))
