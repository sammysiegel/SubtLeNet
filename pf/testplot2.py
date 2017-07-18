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

top_4 = make_coll('/home/snarayan/hscratch/baconarrays/v4/RSGluonToTT_*_4_XXXX.npy') # T
qcd_0 = make_coll('/home/snarayan/hscratch/baconarrays/v4/QCD_*_0_XXXX.npy') # q/g

bins = np.arange(0,1.1,0.05)
h_top = top_4.draw_singletons([('tau32', bins)])['tau32']
h_qcd = qcd_0.draw_singletons([('tau32', bins)])['tau32']

for h in [h_top, h_qcd]:
    h.scale()

p = utils.Plotter()
p.add_hist(h_top, 'top', 'r')
p.add_hist(h_qcd, 'q/g', 'k')
p.plot({'xlabel':r'$\tau_{32}$', 'ylabel':'Probability', 'output':'/home/snarayan/public_html/figs/test/tau32_b'})