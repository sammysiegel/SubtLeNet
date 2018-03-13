#!/usr/bin/env python2.7

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--nepoch',type=int,default=50)
parser.add_argument('--version',type=str,default='4')
parser.add_argument('--trunc',type=int,default=7)
parser.add_argument('--limit',type=int,default=100)
parser.add_argument('--train_baseline',action='store_true')
args = parser.parse_args()

import extra_vars
from subtlenet.models import particles as train
from os import path

train.NEPOCH = args.nepoch
train.VERSION = args.version
data, dims = train.instantiate(args.trunc, args.limit)

clf_gen = train.setup_data(data)

clf = train.build_classifier(dims)

train.train(clf, 'baseline', clf_gen['train'], clf_gen['validation'])
