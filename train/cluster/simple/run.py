#!/usr/bin/env python2.7

from argparse import ArgumentParser
parser = ArgumentParser()
args = parser.parse_args()

from sys import exit
from subtlenet.models import cluster as train
import numpy as np
from subtlenet.utils import mpl, plt
from mpl_toolkits.mplot3d import Axes3D


train.NEPOCH = 10
train.encoded_size=2

dims = train.instantiate('RadialClustering')

gen = train.setup_data(batch_size=100)

clusterer, encoder = train.build_model(dims, w_ae=0.1)

train.train(clusterer, 'cluster', gen['train'], gen['validate'])


plotgen = train.gen(batch_size=1000, label=True)()
i, o, _ = next(plotgen)
i = i[0]
p = clusterer.predict(i)[1]
d = clusterer.predict(i)[0]
e = encoder.predict(i)
print
for b in xrange(1,4):
    print i[b], d[b], p[b], np.argmax(p[b]), o[-1][b], e[b]
    print i[-b], d[-b], p[-b], np.argmax(p[-b]), o[-1][-b], e[-b]
print
w = clusterer.get_weights()[-1][0]
print w

# make 2d plots

plt.clf()
cls = np.argmax(p, axis=-1)
mask = cls == 0
plt.scatter(e[:,0][mask], e[:,1][mask], c='b', alpha=0.5)
mask = cls == 1
plt.scatter(e[:,0][mask], e[:,1][mask], c='r', alpha=0.5)
plt.scatter(w[0], w[1], c='k')

plt.savefig('/home/snarayan/public_html/figs/clustering/encoded.png',bbox_inches='tight',dpi=300)
plt.savefig('/home/snarayan/public_html/figs/clustering/encoded.pdf')

plt.clf()
cls = np.argmax(p, axis=-1)
mask = o[-1] < 0.75
plt.scatter(e[:,0][mask], e[:,1][mask], c='k', alpha=0.5)
mask = o[-1] > 0.75
plt.scatter(e[:,0][mask], e[:,1][mask], c='m', alpha=0.5)

plt.savefig('/home/snarayan/public_html/figs/clustering/encoded_truth.png',bbox_inches='tight',dpi=300)
plt.savefig('/home/snarayan/public_html/figs/clustering/encoded_truth.pdf')

plt.clf()
fig = plt.figure()
ax = Axes3D(fig)
mask = cls == 0
ax.scatter(i[mask,0], i[mask,1], i[mask,2], c='b', alpha=0.5)
mask = cls == 1
ax.scatter(i[mask,0], i[mask,1], i[mask,2], c='r', alpha=0.5)

plt.savefig('/home/snarayan/public_html/figs/clustering/original_clust.png',bbox_inches='tight',dpi=300)
plt.savefig('/home/snarayan/public_html/figs/clustering/original_clust.pdf')

plt.clf()
fig = plt.figure()
ax = Axes3D(fig)
mask = o[-1] < 0.75
ax.scatter(i[mask,0], i[mask,1], i[mask,2], c='k', alpha=0.5)
mask = o[-1] > 0.75
ax.scatter(i[mask,0], i[mask,1], i[mask,2], c='m', alpha=0.5)

plt.savefig('/home/snarayan/public_html/figs/clustering/original.png',bbox_inches='tight',dpi=300)
plt.savefig('/home/snarayan/public_html/figs/clustering/original.pdf')

plt.clf()
fig = plt.figure()
ax = Axes3D(fig)
mask = cls == 0
ax.scatter(d[mask,0], d[mask,1], d[mask,2], c='b', alpha=0.5)
mask = cls == 1
ax.scatter(d[mask,0], d[mask,1], d[mask,2], c='r', alpha=0.5)

plt.savefig('/home/snarayan/public_html/figs/clustering/autoencoded.png',bbox_inches='tight',dpi=300)
plt.savefig('/home/snarayan/public_html/figs/clustering/autoencoded.pdf')

