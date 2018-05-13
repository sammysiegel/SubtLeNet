import numpy as np
import keras.backend as K
import tensorflow as tf

def min_pred(y_true, y_pred):
    # simple loss - just return the minimum of y_pred :)
    min_ = K.min(y_pred, axis=-1)
    return min_

def min_pred_reg(y_true, y_pred):
    min_ = min_pred(y_true, y_pred)
    avg_ = K.sum(y_pred, axis=0) / K.cast(K.shape(y_pred)[0], 'float32')
    var_ = K.var(avg_)
    return min_ + (0.01 * var_)

def huber(y_true, y_pred):
  diff = y_true - y_pred
  sq = 0.5 * K.square(diff)
  lin  = K.abs(diff) - 0.5
  pwise  = K.abs(diff) < 1
  return tf.where(pwise, sq, lin)

class QL(object):
    def __init__(self, q):
        self._q = q
        self.__name__ = 'QL'
    def __call__(self, y_true, y_pred):
        diff = y_true - y_pred
        theta = tf.where(diff < 0, K.ones_like(diff), K.zeros_like(diff))
        return diff * (self._q - theta)

def _weighted_KL(Q_data, P_data, Q_weight=None, P_weight=None):
    if P_weight is not None:
        P = K.sum(P_data * P_weight, axis=0)
    else:
        P = K.sum(P_data, axis=0)
    P = P / K.sum(P, axis=0)
    P = K.clip(P, K.epsilon(), 1)

    if Q_weight is not None:
        Q = K.sum(Q_data * Q_weight, axis=0)
    else:
        Q = K.sum(Q_data, axis=0)
    Q = Q / K.sum(Q, axis=0)
    Q = K.clip(Q, K.epsilon(), 1)

    kl = - K.sum(P * K.log(Q / P), axis=0)
    return kl


def sculpting_kl_penalty(y_true, y_pred):
    # structure of y_pred[i] : [softmax probab of nuis] + [sample weight] + [tag score]
    # structure of y_true[i] : [one-hot vector of nuis] + [sample weight] + [tag truth]
    weight = y_pred[:,-2:-1] * y_true[:,-1:] # set weight to 0 for things that aren't what we want
    tagged_weight = weight * y_pred[:,-1:] # further modify the weight by the prediction
    untagged_weight = weight * (1 - y_pred[:,-1:])

    loss = _weighted_KL(Q_data = y_pred[:,:-2],
                        P_data = y_true[:,:-2], # in practice Q_data = P_data frequently
                        Q_weight = tagged_weight,
                        P_weight = weight)
    loss += _weighted_KL(Q_data = y_pred[:,:-2],
                         P_data = y_true[:,:-2],
                         Q_weight = untagged_weight,
                         P_weight = weight)
    return loss * K.ones_like(weight[:,0]) # make it of dimension (batch_size,)


def emd(p, q, norm=K.abs):
    # p and q are 1-hot vectors of two probability distributions.
    # assume they are normalized to one.
    # typically one is a collection of delta functions.
    # TODO: implement a metric on R
    P = K.cumsum(p, axis=-1)
    Q = K.cumsum(q, axis=-1)

    d = K.sum(norm(P - Q), axis=-1)
    return d


class Binner(object):
    def __init__(self, N):
        self._N = N
        self._ones = K.constant(np.ones(N), shape=(1,N))
        self._bias = K.constant(-1 * np.arange(N))
    @staticmethod
    def _indicator(x):
        return (3 * K.sigmoid(x) * K.sigmoid(1 - x)) - 2
    def __call__(self,x):
        # x should have dimensions (batch_size,)
        xhat = self._N * x                                            # scale into range [0,N]
        xhat = K.dot(K.expand_dims(xhat, axis=-1), self._ones)        # reshape to (batch_size,N)
        xhat = K.bias_add(xhat, self._bias)                           # add a bias vector, same shape
        yhat = Binner._indicator(xhat)                                # make it an indicator of [0,1]
        return K.softmax(yhat)


class DistCompatibility(object):
    @staticmethod
    def _kl(p, q):
        return -K.sum(p * K.log(q / p))
    @staticmethod
    def _loss2(p, q):
        return K.sum(K.abs(K.log(q / p)))
    @staticmethod
    def _emd(p, q):
        P = -K.cumsum(p, axis=-1) # inverted CDF
        Q = K.cumsum(q, axis=-1)  # CDF
        return K.sum(K.abs(K.bias_add(Q, P)))
    def __init__(self, N_bin, N_class, loss='kl'):
        self._N_bin = N_bin
        self._N_class = N_class
        self._binner = Binner(N_bin)
        self.__name__ = type(self).__name__ # ??
        self.loss = getattr(type(self), '_' + loss)
    def __call__(self, y_true, y_pred):
        # structure of y_pred[i] : [tag score] + [class truth]
        # structure of y_true[i] : [sample weight] + [class truth]
        weight = y_true[:,0]
        label = y_true[:,1:]
        x_score = y_pred[:,0]
        batch_size = K.shape(x_score)[0]

        b_score = self._binner(x_score)    # has dimensions (batch_size, _N_bin)
        score = tf.einsum('ij,ik->ikj', b_score, label) # (batch_size, _N_class, _N_bin)
        P = tf.einsum('i,ijk->jk', weight, score) # (_N_class, _N_bin)
        P = P / K.expand_dims(K.sum(P, axis=1)) # normalize along score-axis
        P = K.clip(P, K.epsilon(), 1)

        P0 = P[0,:]
        Q = P[1:,:]

        l = self.loss(P0, Q)

        return l * K.ones_like(weight)
