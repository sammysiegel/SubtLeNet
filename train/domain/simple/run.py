#!/usr/bin/env python2.7

from argparse import ArgumentParser
parser = ArgumentParser()
args = parser.parse_args()

from subtlenet.models import toy as train
import numpy as np

train.NEPOCH = 10
dims = train.instantiate()

clf_gen = train.setup_data(batch_size=32)
adv_gen = train.setup_data(batch_size=32, smear_strengths=[0,0.05,0.1], penalty=True)

clf = train.build_linear(dims)
adv = train.build_continuous_adv(clf, scale=0.1, w_clf=0.1, w_adv=1)

#print 'Pre-rained weights:'
#print clf.get_weights()
#
#train.train(clf, 'baseline', clf_gen['train'], clf_gen['validate'])

clf.set_weights([np.array([[0],[0],[2]], dtype=np.float32), 
                 np.array([0], dtype=np.float32)])

print 'Trained weights:'
print clf.get_weights()

print 'Training the full adversarial stack:'
callback_params = {
        'partial_model' : clf,
        'monitor' : lambda x : 0.1 * x.get('val_y_hat_loss') - x.get('val_adv_loss'), # semi-arbitrary
        }
train.train(adv, 'classified', adv_gen['train'], adv_gen['validate'], callback_params)

print 'Decorrelated weights:'
print clf.get_weights()
