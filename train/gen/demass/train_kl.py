#!/usr/bin/env python2.7

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--nepoch',type=int,default=20)
parser.add_argument('--version',type=int,default=4)
parser.add_argument('--loss',type=str,default='kl')
args = parser.parse_args()

from os import path
import extra_vars
from subtlenet.models import singletons as train

train.NEPOCH = args.nepoch
train.VERSION = args.version
data, dims = train.instantiate()

kl_gen = train.setup_data(data, opts={'kl_decorr_mass':True})
clf = train.build_classifier(dims)

# preload = '%s/%s/classifier_best.h5'%(train.MODELDIR, train._APOSTLE)
# if path.isfile(preload):
#     print 'Pre-loading weights from',preload
#     tmp_ = train.load_model(preload)
#     clf.set_weights(tmp_.get_weights())

kl = train.build_kl_mass(clf)

callback_params = {
        'partial_model' : clf,
        'monitor' : lambda x : x.get('val_y_hat_loss') 
        }
train.train(kl, args.loss+'_decorr_var', kl_gen['train'], kl_gen['validation'], callback_params)
