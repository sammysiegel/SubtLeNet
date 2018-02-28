#!/usr/bin/env python2.7

from subtlenet.train import particles as train
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--nepoch',type=int,default=50)
parser.add_argument('--version',type=int,default=4)
parser.add_argument('--trunc',type=int,default=4)
parser.add_argument('--limit',type=int,default=50)
args = parser.parse_args()

train.NEPOCH = args.nepoch
train.VERSION = args.version
data, dims = train.instantiate(args.trunc, args.limit)

clf_gen = train.setup_data(data)

clf = train.build_classifier(dims)

train.train(clf, 'classifier', clf_gen['train'], clf_gen['validation'])
