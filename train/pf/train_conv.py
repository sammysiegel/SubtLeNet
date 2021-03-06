#!/usr/local/bin/python2.7

from sys import exit 
from os import environ, system
environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
environ["CUDA_VISIBLE_DEVICES"] = ""

environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
import utils
import signal

from keras.layers import Input, Dense, Dropout, Activation, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, LSTM, Convolution1D, MaxPooling1D, MaxPooling1D, Flatten, BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.models import Model 
from keras.callbacks import ModelCheckpoint, LambdaCallback, TensorBoard
from keras.optimizers import Adam, SGD
from keras.utils import np_utils
import obj 
obj.DEBUG = False
# obj.weights_scale = np.array([0, 1, 0, 0.5], dtype=np.float32)
# obj.truth = 'resonanceType'
# obj.n_truth = 5

if obj.truth == 'resonanceType':
    sig_truth = 4
else:
    sig_truth = 3

from keras import backend as K
K.set_image_data_format('channels_last')


def make_coll(fpath):
    coll = obj.PFSVCollection()
    coll.add_categories(['singletons', 'inclusive'], fpath) 
    return coll 

top_4 = make_coll('/home/snarayan/hscratch/baconarrays/v8_repro/PARTITION/RSGluonToTT_3_*_CATEGORY.npy') # T
qcd_0 = make_coll('/home/snarayan/hscratch/baconarrays/v8_repro/PARTITION/QCD_1_*_CATEGORY.npy') # T

data = [top_4, qcd_0]

# preload some data just to get the dimensions
data[0].objects['train']['inclusive'].load(memory=False)
dims = data[0].objects['train']['inclusive'].data.shape 


# inclusive layer
input_inclusive = Input(shape=(dims[1], dims[2]), name='input_inclusive')

norm = BatchNormalization(momentum=0.6, name='input_inclusive_bnorm')(input_inclusive)

conv = Convolution1D(32, 1, activation='relu', name='conv0', kernel_initializer='lecun_uniform')(norm)
norm = BatchNormalization(momentum=0.6, name='bnorm0')(conv)
# drop = Dropout(0.1, name='drop0')(norm)

conv = Convolution1D(16, 1, activation='relu', name='conv1', kernel_initializer='lecun_uniform')(norm)
norm = BatchNormalization(momentum=0.6, name='bnorm1')(conv)
# drop = Dropout(0.1, name='drop1')(norm)

conv = Convolution1D(5, 1, activation='relu', name='conv2', kernel_initializer='lecun_uniform')(norm)

lstm_inclusive = LSTM(100, go_backwards=True, implementation=2, name='lstm')(conv)
norm = BatchNormalization(momentum=0.6, name='lstm_norm')(lstm_inclusive)
# drop = Dropout(0.1, name='lstm_drop')(norm)


dense = Dense(100, activation='relu',name='lstmdense',kernel_initializer='lecun_uniform')(norm)
norm = BatchNormalization(momentum=0.6,name='lstmdensenorm')(dense)
for i in xrange(5):
    dense = Dense(50, activation='relu',name='dense%i'%i)(norm)
    norm = BatchNormalization(momentum=0.6,name='densenorm%i'%i)(dense)
output_p = Dense(obj.n_truth, activation='softmax')(norm)


# model = Model(inputs=[input_charged, input_inclusive, input_sv], outputs=[output_p, output_b])
model = Model(inputs=input_inclusive, outputs=output_p)
model.compile(optimizer=Adam(lr=0.0005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print model.summary()

train_generator = obj.generatePF(data, partition='train', batch=500, mask=True)
validation_generator = obj.generatePF(data, partition='validate', batch=100, mask=True)
test_generator = obj.generatePF(data, partition='validate', batch=1000, mask=True)
test_i, test_o, test_w = next(test_generator)
pred = model.predict(test_i)
print test_o[:5]
print pred[:5]
print test_o[-5:]
print pred[-5:]

# x, y, w = next(train_generator)
# model.fit(x, y, sample_weight=w, epochs=1, batch_size=32, verbose=1)

system('mv train3.log train3.log.old')
flog = open('train3.log','w')
callback = LambdaCallback(
    on_batch_end=lambda batch, logs: flog.write('batch=%i,acc=%f,loss=%f\n'%(batch,logs['acc'],logs['loss']))
    )

tb = TensorBoard(
    log_dir = './logs',
    write_graph = True,
    write_images = True
    )

def save_model(signal=None, frame=None):
    print 'Saving model'
    model.save('model3_pf100.h5')
    print 'Saved model'
    flog.close()
    exit(1)

# ctrl+C now triggers a graceful exit
signal.signal(signal.SIGINT, save_model)

try:
  model.fit_generator(train_generator, 
                      steps_per_epoch=1000, 
                      # steps_per_epoch=10, 
                      callbacks=[callback, tb],
                      epochs=1, 
                      validation_data=validation_generator, 
                      validation_steps=100)
except StopIteration:
  pass

save_model()


pred = model.predict(test_i)
print test_o[:5]
print pred[:5]
print test_o[-5:]
print pred[-5:]





# model.fit(x['train'], y['train'], sample_weight=w['train'],
#           batch_size=30, epochs=1, verbose=1,
#           validation_data=(x['val'],y['val'],w['val']), 
#           shuffle=True)


# y_pred = model.predict(x['test'])
# test_accuracy = np.sum(
#                     (np.argmax(y['test'], axis=1)==np.argmax(y_pred, axis=1))
#                 )/float(x['test'].shape[0])

# print 'NN accuracy = %.3g'%(test_accuracy)

# score = model.evaluate(x['test'], y['test'], batch_size=32, verbose=1, sample_weight=w['test'])

# print '' 
# print 'NN score =',score
