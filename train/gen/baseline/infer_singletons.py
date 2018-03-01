#!/usr/bin/env python2.7

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--h5',type=str)
parser.add_argument('--name',type=str,default=None)
args = parser.parse_args()

import imp

workdir = '/'.join(args.h5.split('/')[:-1])
if not args.name:
    args.name = args.h5.split('/')[-1].replace('.h5','')

imp.load_source('setup', workdir + '/setup.py')
from subtlenet import config

from subtlenet.models import singletons as train
train.infer(args.h5, args.name)
