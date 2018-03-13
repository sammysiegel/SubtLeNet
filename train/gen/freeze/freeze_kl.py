#!/usr/bin/env python2.7

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--nepoch',type=int,default=30)
parser.add_argument('--version',type=str,default='4_kl')
parser.add_argument('--trunc',type=int,default=7)
parser.add_argument('--limit',type=int,default=100)
parser.add_argument('--adv',type=str,default=None)
parser.add_argument('--train_baseline',action='store_true')
parser.add_argument('--window',action='store_true')
parser.add_argument('--reshape',action='store_true')
parser.add_argument('--suffix',type=str,default='')
args = parser.parse_args()

import extra_vars
from subtlenet.models import particles as train
from os import path
from sys import exit

train.NEPOCH = args.nepoch
train.VERSION = args.version
data, dims = train.instantiate(args.trunc, args.limit)

clf_gen = train.setup_data(data)
adv_gen = train.setup_kl_data(data, window=args.window, reshape=args.reshape)


opts = {
        'loss' : args.adv,
        'w_clf' : 0.0001, # used to be 0.001, 0.005
        'w_adv' : 1,     # used to be 1
        }

clf = train.build_classifier(dims)

preload = '%s/%s/baseline_best.h5'%(train.MODELDIR, train._APOSTLE.replace('kl','freeze'))
if path.isfile(preload):
    print 'Pre-loading weights from',preload
    tmp_ = train.load_model(preload)
    clf.set_weights(tmp_.get_weights())
if args.train_baseline or not(path.isfile(preload)):
    train.train(clf, 'baseline', clf_gen['train'], clf_gen['validation'])

frozen_clf = train.partial_freeze(clf, train.compilation_args('classifier'))
adv = train.build_kl_mass(clf=frozen_clf, **opts)

print 'Training the full KL stack:'
callback_params = {
        'partial_model' : clf,
        'monitor' : 'val_loss',
        #'monitor' : lambda x : x.get('val_y_hat_loss') - x.get('val_adv0_loss') - x.get('val_adv1_loss'), # semi-arbitrary
        #'monitor' : lambda x : x.get('val_y_hat_loss') - x.get('val_adv_loss'), # semi-arbitrary
        }
train.train(adv, 'kl'+args.suffix, adv_gen['train'], adv_gen['validation'], callback_params)
