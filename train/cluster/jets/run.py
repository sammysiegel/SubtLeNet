#!/usr/bin/env python2.7

from argparse import ArgumentParser
parser = ArgumentParser()
args = parser.parse_args()

from sys import exit
from os import environ
from subtlenet.models import cluster_jet as train
import numpy as np
from subtlenet import utils, config
from subtlenet.utils import mpl, plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde

from subtlenet import config


train.NEPOCH = 10

data, dims = train.instantiate()

gen = train.setup_data(data)

clusterer, encoder = train.build_model(dims, w_ae=1, w_cl=10)

train.train(clusterer, 'cluster', gen['train'], gen['validate'])


ihc = {}
ohc = {}
ec = []
masks = {(x,y):[] for x in ['s','b'] for y in [0,1]}

plotgen = gen['test'] 
plotdir = environ['FIGSDIR'] 
centers = clusterer.get_weights()[-1][0]
NTEST = 20
for itest in xrange(NTEST):
    i, o_sum, w = next(plotgen)
    i = i[0]
    o = o_sum[0]
    lbl = o_sum[-1]
    w = w[0]
    p = clusterer.predict(i)[1]
    d = clusterer.predict(i)[0]
    e = encoder.predict(i)
    cls = np.argmax(p, axis=-1)

    m = {}
    m['s'] = lbl > 2
    m['b'] = ~m['s']
    m[0] = cls == 0
    m[1] = ~m[0]

    for x in ['s','b']:
        for y in [0,1]:
            m[(x,y)] = np.logical_and(m[x], m[y])

    masks[('s',0)].append(m[('s',0)])
    masks[('s',1)].append(m[('s',1)])
    masks[('b',0)].append(m[('b',0)])
    masks[('b',1)].append(m[('b',1)])
    ec.append(e)

    def make_ihist(idx, mask, d):
        if (idx, mask) not in d:
            h = utils.NH1(bins=np.linspace(-3,3,20))
            d[(idx,mask)] = h
        else:
            h = d[(idx,mask)]
        h.fill_array(i[m[mask],idx], w[m[mask]])
        h.scale()
        return h

    def make_ohist(idx, mask, d):
        if (idx, mask) not in d:
            h = utils.NH1(bins=np.linspace(-3,3,20))
            d[(idx,mask)] = h
        else:
            h = d[(idx,mask)]
        h.fill_array(o[m[mask],idx], w[m[mask]])
        h.scale()
        return h


    for vidx in xrange(len(config.gen_default_variables)):
        v = i[:,vidx]

        ih_s = make_ihist(vidx, 's', ihc)
        ih_b = make_ihist(vidx, 'b', ihc)
        ih_0 = make_ihist(vidx, 0, ihc)
        ih_1 = make_ihist(vidx, 1, ihc)

        oh_s = make_ohist(vidx, 's', ohc)
        oh_b = make_ohist(vidx, 'b', ohc)
        oh_0 = make_ohist(vidx, 0, ohc)
        oh_1 = make_ohist(vidx, 1, ohc)

        if itest == NTEST - 1:
            utils.p.clear()
            utils.p.add_hist(ih_s, 'Signal', 'm')
            utils.p.add_hist(ih_b, 'Bkg', 'k')
            utils.p.add_hist(ih_0, 'Class 0', 'b')
            utils.p.add_hist(ih_1, 'Class 1', 'r')
            utils.p.plot(xlabel=config.gen_default_variables[vidx],
                         output = plotdir + '/%i_input'%vidx)

            utils.p.clear()
            utils.p.add_hist(oh_s, 'Signal', 'm')
            utils.p.add_hist(oh_b, 'Bkg', 'k')
            utils.p.add_hist(oh_0, 'Class 0', 'b')
            utils.p.add_hist(oh_1, 'Class 1', 'r')
            utils.p.plot(xlabel=config.gen_default_variables[vidx],
                         output = plotdir + '/%i_output'%vidx)


e = np.concatenate(ec, axis=0)
for k,v in masks.iteritems():
    masks[k] = np.concatenate(v, axis=0)

sig_like = 1 if np.sum(masks['s',1]) > np.sum(masks['s',0]) else 0

def plot_encoded(x, y):
    plt.clf()
    plt.scatter(e[masks[('s',sig_like)],x], e[masks[('s',sig_like)],y], c=utils.default_colors[0], alpha=0.5, label='S %i'%sig_like)
    plt.scatter(e[masks[('s',1-sig_like)],x], e[masks[('s',1-sig_like)],y], c=utils.default_colors[1], alpha=0.5, label='S %i'%(1-sig_like))
    plt.scatter(e[masks[('b',1-sig_like)],x], e[masks[('b',1-sig_like)],y], c=utils.default_colors[2], alpha=0.5, label='B %i'%(1-sig_like))
    plt.scatter(e[masks[('b',sig_like)],x], e[masks[('b',sig_like)],y], c=utils.default_colors[3], alpha=0.5, label='B %i'%sig_like)
    plt.scatter(centers[x], centers[y], c='k', label='Centers')
    plt.legend(loc=0)
    plt.savefig(plotdir + '/encoded%i%i.png'%(x,y),bbox_inches='tight',dpi=150)
    plt.savefig(plotdir + '/encoded%i%i.pdf'%(x,y))

for x in xrange(4):
    for y in xrange(x):
        plot_encoded(x, y)
