#!/usr/bin/env python2.7

from argparse import ArgumentParser
parser = ArgumentParser()
args = parser.parse_args()

from sys import exit
from os import environ
from subtlenet.models import exc as train
import numpy as np
from subtlenet import utils, config
from subtlenet.backend import obj
from subtlenet.utils import mpl, plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde

train.NEPOCH = 20
#obj._RANDOMIZE = False
FIGSDIR = environ['FIGSDIR']

data, dims = train.instantiate()

ks = [1,2,3,4,5]
colors = ['b', 'm', 'r', 'g', 'xkcd:orange', 'xkcd:sky blue']
gen = train.setup_data(data, ks=ks)

clusterer = train.build_simple(dims, ks)

x, y, pt = next(gen['train'])

for k in ks:
    l = clusterer.get_layer('kmeans%i'%k)
    w = l.get_weights()[0]
    angles = np.linspace(0, 2 * np.pi, k + 1)[:-1] + 1
    etas = 3.5 * np.cos(angles)
    phis = np.pi + 2 * np.sin(angles)
    l.set_weights([np.hstack([etas, phis]).reshape((1,2,k))])

#clusterer.get_layer('kmeans2').set_weights(
#        [np.array(
#            [
#                [
#                    [x[0][0][0],x[0][1][0]],
#                    [x[0][0][1],x[0][1][1]]
#                ]
#            ], 
#          dtype=np.float32)])

#print clusterer.get_weights()[-1][0]

def plot(x, weighting, name, centers_):
    plt.clf()
    idx = weighting.argsort()
    eta = x[idx,0]; phi = (x[idx,1] % (2 * np.pi))
    w=weighting[idx]
    plt.scatter(phi, eta, c=w, cmap=plt.cm.Blues, alpha=0.8)
    plt.colorbar(format='%.1f')
    for N in ks:
        centers = centers_[N]
        eta = centers[0]; phi = (centers[1] % (2 * np.pi))
        print zip(phi, eta)
        plt.scatter(phi, eta, marker='+', c=colors[N], label='Centers %i'%N)
    plt.ylim(-5,5); plt.xlim(0,2*3.14159)
    plt.legend(loc=0)
    print 'creating',name
    plt.savefig(FIGSDIR + '/%s.png'%name,  dpi=150)
    plt.savefig(FIGSDIR + '/%s.pdf'%name)

def plot_d(x, p, name, centers):
    plt.clf()
    # print p.shape
    cls = np.argmin(p, axis=-1)
    eta = x[:,0]; phi = (x[:,1] % (2 * np.pi))
    plt.scatter(phi, eta, c=cls, cmap=plt.cm.Accent, alpha=0.8)
    plt.colorbar(format='%.1f')
    eta = centers[0]; phi = (centers[1] % (2 * np.pi))
    plt.scatter(phi, eta, marker='+', c='r', label='Centers')
    plt.ylim(-5,5); plt.xlim(0,2*3.14159)
    plt.legend(loc=0)
    print 'creating',name
    plt.savefig(FIGSDIR + '/%s.png'%name,  dpi=150)
    plt.savefig(FIGSDIR + '/%s.pdf'%name)

for i in xrange(4):
    d = train.cluster(clusterer, data=(x, y, pt), skip_train=(i==0))
    n = (i) * train.NEPOCH

    centers = {N:d['weights']['kmeans%i'%N][0][0] for N in ks}

    plot(d['x'], d['pt'], '%i_pt'%n, centers)

    for k in [3, 4, 5]:
        plot_d(d['x'], d['distances'][k-1], '%i_d_%i'%(n, k), centers[k])

    # distances = np.min(d['distances'], axis=1)
    # plot(d['x'], 1./distances, '%i_d'%n, centers)
