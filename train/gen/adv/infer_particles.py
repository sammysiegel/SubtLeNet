#!/usr/bin/env python2.7

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--h5',type=str)
parser.add_argument('--name',type=str,default=None)
args = parser.parse_args()

import extra_vars
from subtlenet.models import particles as train
import imp

workdir = '/'.join(args.h5.split('/')[:-1])
if not args.name:
    args.name = args.h5.split('/')[-1].replace('.h5','')

imp.load_source('setup', workdir + '/setup.py')

train.infer(args.h5, args.name)
