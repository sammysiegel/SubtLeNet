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

n_batches = 300
partition = 'test'

p = utils.Plotter()
r = utils.Roccer()

OUTPUT = figsdir + '/' 
system('mkdir -p %s'%OUTPUT)

components = [
              'singletons',
              ]

colls = {
    't' : make_coll(basedir + '/PARTITION/Top_*_CATEGORY.npy',categories=components),
    'q' : make_coll(basedir + '/PARTITION/QCD_*_CATEGORY.npy',categories=components),
}


def access(data, v):
    return data['singletons'][:,config.gen_singletons[v]]

def makebins(lo, hi, w):
    return np.linspace(lo, hi, int((hi-lo)/w))

f_vars = {
    'msd'     : (lambda x : access(x, 'msd'), makebins(0.,400.,10.), r'$m_\mathrm{SD}$ [GeV]'),
}


# unmasked first
hists = {}
for k,v in colls.iteritems():
    hists[k] = v.draw(components=components,
                      f_vars=f_vars,
                      n_batches=n_batches, partition=partition)
    hists[k]['msd'].scale()


hratio = hists['t']['msd'].clone()
hratio.divide(hists['q']['msd'])

hratio.save('mass_scale.npy')

p.clear()
p.add_hist(hratio)
p.plot(output=OUTPUT+'mass_scale', xlabel=r'$m_\mathrm{SD}$ [GeV]',logy=True)

