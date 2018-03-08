#!/usr/bin/env python2.7

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--nepoch',type=int,default=50)
parser.add_argument('--version',type=int,default=4)
args = parser.parse_args()

import extra_vars
from subtlenet.models import singletons as train

train.NEPOCH = args.nepoch
train.VERSION = args.version
data, dims = train.instantiate()

clf_gen = train.setup_data(data)
clf = train.build_classifier(dims)
train.train(clf, 'classifier', clf_gen['train'], clf_gen['validation'])
