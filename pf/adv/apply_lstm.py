#!/usr/local/bin/python2.7

from sys import exit 
from os import environ, system
environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import utils


from keras.models import Model, load_model 
from keras.utils import np_utils
import obj 
import config 
# obj.DEBUG = True
# obj.truth = 'resonanceType'
# config.n_truth = 5

classifier = load_model('classifier.h5')

n_batches = 200
partition = 'test'

from keras import backend as K
K.set_image_data_format('channels_last')


def make_coll(fpath):
    coll = obj.PFSVCollection()
    coll.add_categories(['singletons', 'inclusive'], fpath) 
    return coll 

top_4 = make_coll('/home/snarayan/scratch5/baconarrays/v10_repro/PARTITION/ZprimeToTTJet_3_*_CATEGORY.npy') # T
qcd_0 = make_coll('/home/snarayan/scratch5/baconarrays/v10_repro/PARTITION/QCD_1_*_CATEGORY.npy') # T


# run DNN
def predict(data, model):
    return model.predict([data['inclusive']])[:,config.n_truth-1]

f_vars = {
  'tau32' : (lambda x : x['singletons'][:,obj.singletons['tau32']], np.arange(0,1.2,0.01)),
  'partonM' : (lambda x : x['singletons'][:,obj.singletons['partonM']], np.arange(0,400,5)),
  'msd'   : (lambda x : x['singletons'][:,obj.singletons['msd']], np.arange(0.,400.,5.)),
  'pt'    : (lambda x : x['singletons'][:,obj.singletons['pt']], np.arange(250.,1000.,50.)),
  'classifier'   : (lambda x : predict(x, classifier), np.arange(0,1.2,0.001)),
}

OUTPUT = '/home/snarayan/public_html/figs/badnet/adversarial_pf/'
system('mkdir -p '+OUTPUT)

p = utils.Plotter()
r = utils.Roccer()


# unmasked first
hists_top = top_4.draw(components=['singletons', 'inclusive'],
                       f_vars=f_vars, n_batches=n_batches)
hists_qcd = qcd_0.draw(components=['singletons', 'inclusive'],
                       f_vars=f_vars, n_batches=n_batches)

for k in hists_top:
    htop = hists_top[k]
    hqcd = hists_qcd[k]
    htop.scale() 
    hqcd.scale()
    p.clear()
    p.add_hist(htop, '3-prong', 'r')
    p.add_hist(hqcd, '1-prong', 'k')
    p.plot({'output':OUTPUT+'unmasked_'+k})

r.clear()
r.add_vars(hists_top,
           hists_qcd,
           {'tau32':r'$\tau_{32}$', 'classifier':'classifier', 'msd':r'$m_{SD}$'},
           {'tau32':'k', 'classifier':'r',  'msd':'b'})
r.plot({'output':OUTPUT+'unmasked_roc'})


# get the cuts
threshold = 0.95
h = hists_qcd['classifier']
classifier_cut = 0
for ib in xrange(h.bins.shape[0]):
    frac = h.integral(lo=0, hi=ib) / h.integral()
    if frac >= threshold:
        classifier_cut = h.bins[ib]
        break

h = hists_qcd['tau32']
tau32_cut = 0
for ib in xrange(h.bins.shape[0]):
    frac = h.integral(lo=(h.bins.shape[0]-ib), hi=None) / h.integral()
    if frac >= threshold:
        tau32_cut = h.bins[h.bins.shape[0]-ib]
        break

print 'classifier', classifier_cut
print 'tau32', tau32_cut

# mask pretrain
def f_mask(data, model, cut):
    return predict(data, model) > cut

hists_top = top_4.draw(components=['singletons', 'inclusive'],
                       f_vars=f_vars, n_batches=n_batches, 
                       f_mask = lambda x : f_mask(x, classifier, classifier_cut))
hists_qcd = qcd_0.draw(components=['singletons', 'inclusive'],
                       f_vars=f_vars, n_batches=n_batches, 
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
