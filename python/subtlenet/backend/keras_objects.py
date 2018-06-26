from keras.layers import Input, Dense, Dropout, concatenate, LSTM, \
        BatchNormalization, Conv1D, concatenate, CuDNNGRU, GRU, CuDNNLSTM, Flatten, \
        Lambda, LeakyReLU, multiply

from keras.optimizers import Adam, RMSprop, Adadelta, Adagrad, Nadam, SGD

from keras.regularizers import L1L2

import keras.backend as K

from keras.losses import categorical_crossentropy, mse

from keras.utils import np_utils

def AMSgrad(*args, **kwargs):
    return Adam(*args, amsgrad=True, **kwargs)
