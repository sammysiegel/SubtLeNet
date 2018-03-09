#!/usr/bin/env python2.7

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--nepoch',type=int,default=50)
parser.add_argument('--version',type=int,default=4)
args = parser.parse_args()

from os import path
import extra_vars
from subtlenet.models import singletons as train

train.NEPOCH = args.nepoch
train.VERSION = args.version
data, dims = train.instantiate()

n_classes = len(data) - 1
# n_classes = 1 # override

adv_gen = train.setup_data(data, opts={'decorr_label':n_classes})
clf = train.build_classifier(dims)

# preload = '%s/%s/classifier_best.h5'%(train.MODELDIR, train._APOSTLE)
# if path.isfile(preload):
#     print 'Pre-loading weights from',preload
#     tmp_ = train.load_model(preload)
#     clf.set_weights(tmp_.get_weights())

adv = train.build_adversary(clf, 'categorical_crossentropy', 1, 0.00003, 10., N = n_classes)

callback_params = {
        'partial_model' : clf,
        'monitor' : lambda x : x.get('val_y_hat_loss'), # semi-arbitrary
        }
train.train(adv, 'adv_decorr_var', adv_gen['train'], adv_gen['validation'], callback_params)
