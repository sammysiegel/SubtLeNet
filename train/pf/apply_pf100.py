#!/usr/local/bin/python2.7

from sys import exit 
from os import environ, system
environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import utils


from keras.layers import Input, Dense, Dropout, Activation, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, LSTM, Convolution1D, MaxPooling1D, MaxPooling1D
from keras.models import Model, load_model 
from keras.callbacks import ModelCheckpoint 
from keras.optimizers import Adam 
from keras.utils import np_utils
import obj 
# obj.DEBUG = True
# obj.truth = 'resonanceType'
# obj.n_truth = 5

model = load_model('model3_pf100.h5')
print model.summary()

from keras import backend as K
K.set_image_data_format('channels_last')


def make_coll(fpath):
    coll = obj.PFSVCollection()
    coll.add_categories(['singletons', 'inclusive'], fpath) 
    return coll 

top_4 = make_coll('/home/snarayan/hscratch/baconarrays/v8_repro/PARTITION/RSGluonToTT_3_*_CATEGORY.npy') # T
qcd_0 = make_coll('/home/snarayan/hscratch/baconarrays/v8_repro/PARTITION/QCD_1_*_CATEGORY.npy') # T

data = [top_4, qcd_0]

hists_top = {
  'tau32' : utils.NH1(np.arange(0,1.2,0.05)),
  'msd' : utils.NH1(np.arange(0.,400.,10.)),
  'dnn' : utils.NH1(np.arange(0,1.2,0.01))
}
hists_qcd = {
  'tau32' : utils.NH1(np.arange(0,1.2,0.05)),
  'msd' : utils.NH1(np.arange(0.,400.,10.)),
  'dnn' : utils.NH1(np.arange(0,1.2,0.01))
}

hists_top['tau32'] = top_4.draw_singletons([('tau32', hists_top['tau32'].bins)], partition='test')['tau32']
hists_qcd['tau32'] = qcd_0.draw_singletons([('tau32', hists_qcd['tau32'].bins)], partition='test')['tau32']

top_4.refresh(partitions=['test'])
qcd_0.refresh(partitions=['test'])

hists_top['msd'] = top_4.draw_singletons([('msd', hists_top['msd'].bins)], partition='test')['msd']
hists_qcd['msd'] = qcd_0.draw_singletons([('msd', hists_qcd['msd'].bins)], partition='test')['msd']

top_4.refresh(partitions=['test'])
qcd_0.refresh(partitions=['test'])

# test_generator = obj.generatePFSV([top_4], partition='test', batch=100)
test_generator = obj.generatePF(data, partition='test', batch=10000, repartition=False)

while True:
    try:
        i, o, w = next(test_generator)
        pred = model.predict(i)[:,obj.n_truth-1]
        o = np.array(o)
        mask_signal = (o[:,obj.n_truth-1] == 1)
        mask_background = np.logical_not(mask_signal)
        hists_top['dnn'].fill_array(pred[mask_signal], w[mask_signal])
        hists_qcd['dnn'].fill_array(pred[mask_background], w[mask_background])
    except StopIteration:
        break

OUTPUT = '/home/snarayan/public_html/figs/testplots/test_pf100/'
system('mkdir -p '+OUTPUT)

# qcd_4.draw_singletons([('tau32', hists_qcd['tau32'].bins)], partition='test')


for h in [hists_top, hists_qcd]:
    for hh in h.values():
        hh.scale()

p = utils.Plotter()
p.add_hist(hists_top['dnn'], 'top', 'r')
p.add_hist(hists_qcd['dnn'], 'q/g', 'k')
p.plot({'output':OUTPUT+'dnn'})

p.clear()
p.add_hist(hists_top['tau32'], 'top', 'r')
p.add_hist(hists_qcd['tau32'], 'q/g', 'k')
p.plot({'output':OUTPUT+'tau32'})


p.clear()
p.add_hist(hists_top['msd'], 'top', 'r')
p.add_hist(hists_qcd['msd'], 'q/g', 'k')
p.plot({'output':OUTPUT+'msd'})

r = utils.Roccer()
r.addROCs(hists_top,
          hists_qcd,
          {'tau32':r'$\tau_{32}$', 'dnn':'DNN', 'msd':r'$m_{SD}$'},
          {'tau32':'k', 'dnn':'r', 'msd':'b'})
r.plotROCs({'output':OUTPUT+'roc'})
