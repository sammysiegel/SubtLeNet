#!/usr/local/bin/python2.7

from sys import exit 
from os import environ
environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import utils

import obj 
obj.DEBUG = True


def make_coll(fpath):
    coll = obj.PFSVCollection()
    coll.add_classes(['singletons', 'charged', 'inclusive', 'sv'], fpath) 
    return coll 

top_4 = make_coll('/home/snarayan/hscratch/baconarrays/v6/RSGluonToTT_*_4_XXXX.npy') # T
qcd_0 = make_coll('/home/snarayan/hscratch/baconarrays/v6/QCD_*_0_XXXX.npy') # q/g
# qcd_0 = make_coll('/home/snarayan/hscratch/baconarrays/v6/QCD_*_0_XXXX.npy') # q/g


bins = {}
bins['tau32'] = np.arange(0,1.1,0.05)
bins['pt'] = np.arange(200.,2000.,40)
bins['msd'] = np.arange(0,400,10.)

labels = {
  'tau32' : r'$\tau_{32}$',
  'pt' : r'$p_{T} [GeV]$',
  'msd' : r'$m_{SD} [GeV]$',
}

def draw(partition='test'):
  h_top = top_4.draw_singletons(bins.items(), partition=partition)
  h_qcd = qcd_0.draw_singletons(bins.items(), partition=partition)

  for h in [h_top, h_qcd]:
      for v in h.values(): 
        v.scale()

  for k in bins:
    p = utils.Plotter()
    p.add_hist(h_top[k], 'top', 'r')
    p.add_hist(h_qcd[k], 'q/g', 'k')
    p.plot({'xlabel':labels[k], 'ylabel':'Probability', 'output':'/home/snarayan/public_html/figs/%s/'%partition+k})

  if partition=='test':  
      r = utils.Roccer()
      r.addROCs(h_top,h_qcd,labels,colors)
      r.plotROCs({'output':'/home/snarayan/public_html/figs/%s/roc'%partition})      

draw()
draw('validate')
draw('train')
