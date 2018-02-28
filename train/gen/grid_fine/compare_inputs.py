#!/usr/local/bin/python2.7

from sys import exit 
from os import environ, system
environ['KERAS_BACKEND'] = 'tensorflow'
environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
environ["CUDA_VISIBLE_DEVICES"] = ""


import numpy as np

from subtlenet import config, utils
from subtlenet.backend import obj 
from subtlenet.generators.gen import make_coll
import paths 

n_batches = 500
partition = 'test'

p = utils.Plotter()
r = utils.Roccer()

OUTPUT = paths.figsdir + '/' 
system('mkdir -p %s'%OUTPUT)

components = ['singletons','particles']

basedir = '/data/t3serv014/snarayan/deep/v_deepgen_4_0p02_small/'
basedir_gen = '/fastscratch/snarayan/genarrays/v_deepgen_4_small/'


colls = {
    'q' : make_coll(basedir + '/PARTITION/QCD_*_CATEGORY.npy',categories=components),
    'q_gen' : make_coll(basedir_gen + '/PARTITION/QCD_*_CATEGORY.npy',categories=components),
}


# run DNN
def access(data, v):
    return data['singletons'][:,config.gen_singletons[v]]

def div(data, num, den):
    return access(data, num) / np.clip(access(data, den), 0.0001, 999)

f_vars = {
    'nprongs' : (lambda x : access(x, 'nprongs'), np.arange(0,10,0.1), r'$N_\mathrm{prongs}$'),
    'partonm' : (lambda x : access(x, 'partonm'), np.arange(0,250,5), r'Parton mass [Gev]'),
    'x_px'    : (lambda x : x['particles'][:,:10,0], np.arange(-50,50,1), r'Particle $p_x$ [GeV]'),
    'x_py'    : (lambda x : x['particles'][:,:10,1], np.arange(-50,50,1), r'Particle $p_y$ [GeV]'),
    'x_pz'    : (lambda x : x['particles'][:,:10,2], np.arange(-50,300,5), r'Particle $p_z$ [GeV]'),
    'x_px_last'    : (lambda x : x['particles'][:,10:,0], np.arange(-50,50,1), r'Particle $p_x$ [GeV]'),
    'x_py_last'    : (lambda x : x['particles'][:,10:,1], np.arange(-50,50,1), r'Particle $p_y$ [GeV]'),
    'x_pz_last'    : (lambda x : x['particles'][:,10:,2], np.arange(-50,300,5), r'Particle $p_z$ [GeV]'),
}


# unmasked first
hists = {}
for k,v in colls.iteritems():
    hists[k] = v.draw(components=components,
                      f_vars=f_vars,
                      n_batches=n_batches, partition=partition)


for k in hists['q']:
    hqgen = hists['q_gen'][k]
    hq = hists['q'][k]
    for h in [hqgen, hq]:
        h.scale()
    p.clear()
    p.add_hist(hqgen, r'$\delta R = 0$', 'r')
    p.add_hist(hq, r'$\delta R =0.2$', 'k')
    p.plot(output=OUTPUT+k, xlabel=f_vars[k][2])

    p.clear()
    p.add_hist(hqgen, r'$\delta R = 0$', 'r')
    p.add_hist(hq, r'$\delta R =0.2$', 'k')
    p.plot(output=OUTPUT+k+'_logy', xlabel=f_vars[k][2], logy=True)

    

