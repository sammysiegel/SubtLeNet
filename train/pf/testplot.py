#!/usr/local/bin/python2.7

from os import environ
environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import utils


top_arr = np.load('/home/snarayan/hscratch/baconarrays/v3/RSGluonToTT_M_1000_13TeV_pythia8_Output_job3_file0_4_singletons.npy')[:,5]
qcd_arr = np.load('/home/snarayan/hscratch/baconarrays/v3/QCD_Pt_1000to1400_13TeV_pythia8_ext_Output_job101_file0_0_singletons.npy')[:,5]

print top_arr, qcd_arr

top_h = utils.NH1(bins=np.arange(0,1.1,.05))
qcd_h = utils.NH1(bins=np.arange(0,1.1,.05))

top_h.fill_array(top_arr)
qcd_h.fill_array(qcd_arr)

for h in [top_h, qcd_h]:
    h.scale()
    
p = utils.Plotter()
p.add_hist(top_h, 'top', 'r')
p.add_hist(qcd_h, 'q/g', 'k')

p.plot({'xlabel':r'$\tau_{32}$', 'ylabel':'Probability', 'output':'/home/snarayan/public_html/figs/test/tau32'})
