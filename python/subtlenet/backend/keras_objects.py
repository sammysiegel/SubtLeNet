from keras.layers import Input, Dense, Dropout, concatenate, LSTM, \
        BatchNormalization, Conv1D, concatenate, CuDNNGRU, GRU, CuDNNLSTM, Flatten, \
        Lambda

from keras.optimizers import Adam, RMSprop, Adadelta, Adagrad, Nadam

def AMSgrad(*args, **kwargs):
    return Adam(*args, amsgrad=True, **kwargs)
