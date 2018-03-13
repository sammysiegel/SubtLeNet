#!/usr/bin/env python2.7

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--nepoch',type=int,default=50)
parser.add_argument('--version',type=int,default=4)
parser.add_argument('--trunc',type=int,default=4)
parser.add_argument('--limit',type=int,default=50)
parser.add_argument('--train_baseline',action='store_true')
parser.add_argument('--opt',type=str,default='Adam')
args = parser.parse_args()

from subtlenet.models import particles as train
from os import path

train.NEPOCH = args.nepoch
train.VERSION = str(args.version) + '_' + args.opt
train.OPTIMIZER = args.opt
data, dims = train.instantiate(args.trunc, args.limit)

clf_gen = train.setup_data(data)

clf = train.build_classifier(dims)

train.train(clf, 'baseline', clf_gen['train'], clf_gen['validation'])
