from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time

from keras.callbacks import Callback, ModelCheckpoint

class PartialModelCheckpoint(Callback):
    def __init__(self, partial_model, filepath, monitor='val_loss', verbose=0,
                 save_best_only=True, mode='auto'):
        super(PartialModelCheckpoint, self).__init__()
        self.partial_model = partial_model
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.epochs_since_last_save = 0

        if type(monitor) == str:
            self.monitor = lambda x : x.get(monitor)
        else:
            self.monitor = monitor

        if mode not in ['auto', 'min', 'max']:
            mode = 'auto'

        self.monitor_op = np.less
        self.best = np.Inf
        if mode == 'min':
            pass
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        elif type(monitor) == str:
            if 'acc' in monitor or monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        filepath = self.filepath.format(epoch=epoch + 1, **logs)
        if self.save_best_only:
            current = self.monitor(logs)
            if current is None:
                pass
            else:
                if self.monitor_op(current, self.best):
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                              ' saving model to %s'
                              % (epoch + 1, repr(self.monitor), self.best,
                                 current, filepath))
                    self.best = current
                    self.partial_model.save(filepath, overwrite=True)
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: %s did not improve' %
                              (epoch + 1, repr(self.monitor)))
        else:
            if self.verbose > 0:
                print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
            self.partial_model.save(filepath, overwrite=True)

