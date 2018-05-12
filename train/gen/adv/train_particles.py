#!/usr/bin/env python2.7

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--nepoch',type=int,default=40)
parser.add_argument('--version',type=str,default='6')
parser.add_argument('--trunc',type=int,default=4)
parser.add_argument('--limit',type=int,default=50)
parser.add_argument('--adv',type=str,default=None)
parser.add_argument('--train_baseline',action='store_true')
args = parser.parse_args()

import extra_vars
from subtlenet.models import particles as train
from os import path

from subtlenet import config
from subtlenet.backend import obj
# config.DEBUG = True
# obj._RANDOMIZE = False

train.NEPOCH = args.nepoch
train.VERSION = str(args.version) + '_Adam'
#train.OPTIMIZER = 'RMSprop'
data, dims = train.instantiate(args.trunc, args.limit)

clf_gen = train.setup_data(data)
adv_gen = train.setup_data(data, decorr_mass=True)


if args.adv == 'emd':
    opts = {
            'loss' : train.emd,
            'scale' : 0.1,
            'w_clf' : 0.001,
            'w_adv' : 100,
            }
elif args.adv == 'mse':
    opts = {
            'loss' : args.adv,
            'scale' : 0.03,
            'w_clf' : 0.001,
            'w_adv' : 0.1,
            }
else:
    opts = {
            'loss' : args.adv,
            'scale' : 0.1,
            'w_clf' : 0.001,
            'w_adv' : 1,
            }

clf = train.build_classifier(dims)
if args.adv is not None:
    adv = train.build_adversary(clf=clf, **opts)

preload = '%s/%s/baseline_best.h5'%(train.MODELDIR, train._APOSTLE)
if path.isfile(preload):
    print 'Pre-loading weights from',preload
    tmp_ = train.load_model(preload)
    clf.set_weights(tmp_.get_weights())
if args.train_baseline or not(path.isfile(preload)):
    train.train(clf, 'baseline', clf_gen['train'], clf_gen['validation'])

if args.adv:
    print 'Training the full adversarial stack:'
    callback_params = {
            'partial_model' : clf,
            'monitor' : lambda x : opts['w_clf'] * x.get('val_y_hat_loss') - opts['w_adv'] * x.get('val_adv_loss'), # semi-arbitrary
            }
    train.train(adv, args.adv, adv_gen['train'], adv_gen['validation'], callback_params)
