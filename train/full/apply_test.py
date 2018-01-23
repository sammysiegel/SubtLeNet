#!/usr/local/bin/python2.7

from sys import exit 
from os import environ, system
environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import utils


from keras.models import Model, load_model 
from keras.utils import np_utils
import obj 
# obj.DEBUG = True
# obj.truth = 'resonanceType'
# obj.n_truth = 5

pretrained = load_model('pretrained.h5')
regularized = load_model('regularized.h5')

n_batches = 400
partition = 'test'

from keras import backend as K
K.set_image_data_format('channels_last')


def make_coll(fpath):
    coll = obj.PFSVCollection()
    coll.add_categories(['singletons'], fpath) 
    return coll 

top_4 = make_coll('/home/snarayan/scratch5/baconarrays/v9_repro/PARTITION/ZprimeToTTJet_3_*_CATEGORY.npy') # T
qcd_0 = make_coll('/home/snarayan/scratch5/baconarrays/v9_repro/PARTITION/QCD_1_*_CATEGORY.npy') # T


# run DNN
input_indices = [obj.singletons[x] for x in ['msd','tau32','tau21']]
# input_indices = [obj.singletons[x] for x in ['msd','tau32','tau21']]
def predict(data, model):
    return model.predict([data['singletons'][:,input_indices]])[:,obj.n_truth-1]

f_vars = {
  'tau32' : (lambda x : x['singletons'][:,obj.singletons['tau32']], np.arange(0,1.2,0.01)),
  'msd'   : (lambda x : x['singletons'][:,obj.singletons['msd']], np.arange(0.,400.,5.)),
  'pt'    : (lambda x : x['singletons'][:,obj.singletons['pt']], np.arange(250.,1000.,50.)),
  'pretrained'   : (lambda x : predict(x, pretrained), np.arange(0,1.2,0.01)),
  'regularized'   : (lambda x : predict(x, regularized), np.arange(0,1.2,0.01)),
}

OUTPUT = '/home/snarayan/public_html/figs/badnet/adversarial/'
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
           {'tau32':r'$\tau_{32}$', 'pretrained':'pretrained', 'regularized':'regularized', 'msd':r'$m_{SD}$'},
           {'tau32':'k', 'pretrained':'r', 'regularized':'g', 'msd':'b'})
r.plot({'output':OUTPUT+'unmasked_roc'})


# get the cuts
threshold = 0.6
h = hists_qcd['pretrained']
pretrained_cut = 0
for ib in xrange(h.bins.shape[0]):
    frac = h.integral(lo=0, hi=ib) / h.integral()
    if frac >= threshold:
        pretrained_cut = h.bins[ib]
        break
h = hists_qcd['regularized']
regularized_cut = 0
for ib in xrange(h.bins.shape[0]):
    frac = h.integral(lo=0, hi=ib) / h.integral()
    if frac >= threshold:
        regularized_cut = h.bins[ib]
        break

print 'PRETRAINED', pretrained_cut
print 'REGULARIZED', regularized_cut

# mask pretrain
def f_mask(data, model, cut):
    return predict(data, model) > cut

hists_top = top_4.draw(components=['singletons', 'inclusive'],
                       f_vars=f_vars, n_batches=n_batches, 
                       f_mask = lambda x : f_mask(x, pretrained, pretrained_cut))
hists_qcd = qcd_0.draw(components=['singletons', 'inclusive'],
                       f_vars=f_vars, n_batches=n_batches, 
                       f_mask = lambda x : f_mask(x, pretrained, pretrained_cut))

for k in hists_top:
    htop = hists_top[k]
    hqcd = hists_qcd[k]
    htop.scale() 
    hqcd.scale()
    p.clear()
    p.add_hist(htop, '3-prong', 'r')
    p.add_hist(hqcd, '1-prong', 'k')
    p.plot({'output':OUTPUT+'pretrained_'+k})


# mask regularized
hists_top = top_4.draw(components=['singletons', 'inclusive'],
                       f_vars=f_vars, n_batches=n_batches, 
                       f_mask = lambda x : f_mask(x, regularized, regularized_cut))
hists_qcd = qcd_0.draw(components=['singletons', 'inclusive'],
                       f_vars=f_vars, n_batches=n_batches, 
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

