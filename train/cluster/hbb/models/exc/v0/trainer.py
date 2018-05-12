#!/usr/bin/env python2.7

from argparse import ArgumentParser
parser = ArgumentParser()
args = parser.parse_args()

from sys import exit
from os import environ, system
from subtlenet.models import exc as train
import numpy as np
from subtlenet import utils, config
from subtlenet.backend import obj
from subtlenet.utils import mpl, plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import Rbf, interp2d 

train.NEPOCH = 10
train.R = 0.8
obj._RANDOMIZE = False
base_FIGSDIR = environ['FIGSDIR']

data, dims = train.instantiate()

ks = [1,3,5,7,9]
k_map = {x:i for i,x in enumerate(ks)}
colors = ['b', 'm', 'r', 'g', 'xkcd:orange', 'xkcd:sky blue']
gen = train.setup_data(data, ks=ks)

clusterer = train.build_simple(dims, ks)

for NTEST in xrange(10):
    FIGSDIR = base_FIGSDIR + '/%i/'%NTEST
    system('mkdir -p ' + FIGSDIR)

    x, y, pt = next(gen['train'])
    etaphi = [x[0]]
    akt = x[1]

    for k in ks:
        l = clusterer.get_layer('kmeans%i'%k)
        w = l.get_weights()[0]
        angles = np.linspace(0, 2 * np.pi, k + 1)[:-1] + 1
        etas = 3.5 * np.cos(angles)
        phis = (etaphi[0][0,1] + 2 * np.sin(angles)) % (2 * np.pi)
        l.set_weights([np.hstack([etas, phis]).reshape((1,2,k))])

    def plot(x, weighting, name, centers_=None, cmap=plt.cm.viridis):
        plt.clf()
        idx = weighting.argsort()
        eta = x[idx,0]; phi = (x[idx,1] % (2 * np.pi))
        w=weighting[idx]
        plt.scatter(phi, eta, c=w, cmap=cmap, alpha=0.8)
        plt.colorbar(format='%.1f')
        if centers_ is not None:
            for N in ks:
                centers = centers_[N]
                eta = centers[0]; phi = (centers[1] % (2 * np.pi))
#                print zip(phi, eta)
                color = colors[k_map[N]]
                plt.scatter(phi, eta, marker='+', c=color, label='%i clusters'%N)
        plt.ylim(-5,10); plt.xlim(0,2*3.14159)
        plt.legend(loc=1)
        print 'creating',name
        plt.savefig(FIGSDIR + '/%s.png'%name,  dpi=150)
        plt.savefig(FIGSDIR + '/%s.pdf'%name)

    def plot_d(x, p, name, centers, label):
        plt.clf()
        cls = np.argmin(p, axis=-1)
        uncl_mask = (np.array([p[i,c] for i,c in enumerate(cls)]) > 1)
        
        xcls = np.array(cls, copy=True)
        xcls[uncl_mask] = p.shape[1]
        
        eta = x[:,0]; phi = (x[:,1] % (2 * np.pi))
        plt.scatter(phi, eta, c=xcls, cmap=plt.cm.Accent, alpha=0.8)
        plt.colorbar(format='%.1f')
        eta = centers[0]; phi = (centers[1] % (2 * np.pi))
#        print centers 
#        print eta, phi
        plt.scatter(phi, eta, marker='+', c='r', label='Centers')
        plt.ylim(-5,10); plt.xlim(0,2*3.14159)
        plt.text(0.5, 7, label)
        print 'creating',name
        plt.savefig(FIGSDIR + '/%s.png'%name,  dpi=150)
        plt.savefig(FIGSDIR + '/%s.pdf'%name)

    def plot_loss(x, loss, name, centers, label):
        plt.clf()
        idx = loss.argsort()
        eta = x[idx,0]; phi = (x[idx,1] % (2 * np.pi))
        plt.scatter(phi, eta, c=loss[idx], cmap=plt.cm.viridis)
    #    plt.clim(0, 1.5)
        plt.colorbar(format='%.1f')

        eta = centers[0]; phi = (centers[1] % (2 * np.pi))
        plt.scatter(phi, eta, marker='+', c='r', label='Centers')
        plt.ylim(-5,10); plt.xlim(0,2*3.14159)
        plt.text(0.5, 7, label)
        print 'creating',name
        plt.savefig(FIGSDIR + '/%s.png'%name,  dpi=150)
        plt.savefig(FIGSDIR + '/%s.pdf'%name)

    for i in xrange(3):
        d = train.cluster(clusterer, data=(etaphi, y, pt), skip_train=(i==0))
        n = (i) * train.NEPOCH

        centers = {N:d['weights']['kmeans%i'%N][0][0] for N in ks}

        plot(d['x'], d['pt'], '%i_pt'%n, centers)

        plot(d['x'], akt, '%i_eakt'%n, cmap=plt.cm.Accent)

        n_part = np.sum(d['pt'] > 0)

        for k in ks:
            dist = d['distances'][k_map[k]]
            drloss = np.min(dist, axis=-1)
            loss = drloss * d['pt']
            sum_drloss = np.sum(drloss); sum_loss = np.sum(loss)
            label = [
                     r'$N_p = %i$'%(n_part),
                     r'$\sum\min\{\Delta R^2\} = %.3f$'%(sum_drloss), 
                     r'$\sum p_\mathrm{T}^{%i}\min\{\Delta R^2\} = %.3f$'%(train.generator.EXPONENT, sum_loss),
                    ]
            label = '\n'.join(label)
            plot_d(d['x'], dist, '%i_d_%i'%(n, k), centers[k], label)
            plot_loss(d['x'], loss, '%i_loss_%i'%(n, k), centers[k], label)

        # distances = np.min(d['distances'], axis=1)
        # plot(d['x'], 1./distances, '%i_d'%n, centers)
