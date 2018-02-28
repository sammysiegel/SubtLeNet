#!/usr/bin/env python2.7

from subtlenet.models import particles as train
from argparse import ArgumentParser
from os import path

parser = ArgumentParser()
parser.add_argument('--nepoch',type=int,default=50)
parser.add_argument('--version',type=int,default=4)
parser.add_argument('--trunc',type=int,default=4)
parser.add_argument('--limit',type=int,default=50)
parser.add_argument('--adv',type=str,default='categorical_cross_entropy')
parser.add_argument('--train_baseline',action='store_true')
args = parser.parse_args()

train.NEPOCH = args.nepoch
train.VERSION = args.version
data, dims = train.instantiate(args.trunc, args.limit)

clf_gen = train.setup_data(data)
adv_gen = train.setup_adv_data(data)

loss = train.emd if args.adv == 'emd' else args.adv

clf = train.build_classifier(dims)
adv = train.build_adversary(clf=clf,
                            loss=loss,
                            scale=0.5,
                            w_clf=0.05,
                            w_adv=100)

if (not args.train_baseline) and path.isfile('%s/%s/baseline_best.h5'%(train.MODELDIR, train._APOSTLE)):
    tmp_ = train.load_model('%s/%s/baseline_best.h5'%(train.MODELDIR, train._APOSTLE))
    clf.set_weights(tmp_.get_weights())
else:
    train.train(clf, 'classifier', clf_gen['train'], clf_gen['validation'])

callbacks = [train.PartialModelCheckpoint(
                    clf,
                    '%s/%s/%s_clf_best.h5'%(train.MODELDIR, train._APOSTLE, args.adv),
                    save_best_only=True,
                    verbose=True,
                )]

train.train(adv, args.adv, adv_gen['train'], adv_gen['validation'], callbacks)
