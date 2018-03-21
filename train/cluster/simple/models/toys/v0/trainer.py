#!/usr/bin/env python2.7

from argparse import ArgumentParser
parser = ArgumentParser()
args = parser.parse_args()

from sys import exit
from subtlenet.models import toy as train
import numpy as np

train.NEPOCH = 10
dims = train.instantiate()

gen = train.setup_data(batch_size=32)

clusterer, autoencoder = train.build_model(dims)

train.train(clusterer, 'cluster', gen['train'], gen['validate'])

i, o, _ = next(gen['test'])
p = clusterer.predict(i)

for b in xrange(32):
    print i[0][b], p[0][b], p[1][b]

