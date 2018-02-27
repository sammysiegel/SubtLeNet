import numpy as np
import keras.backend as K

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
