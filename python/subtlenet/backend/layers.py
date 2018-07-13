from __future__ import division

import numpy as np
import keras.backend as K
from keras.engine import Layer, InputSpec
from keras.layers import RNN, Dense
from keras import activations, initializers, regularizers, constraints
import tensorflow as tf
from .. import config


def phi_diff(phi1, phi2):
    phi1 = tf.mod(phi1, 2*np.pi)
    phi2 = tf.mod(phi2, 2*np.pi)
    diff = phi2 - phi1
    return tf.minimum(diff, 2*np.pi - diff)
    # x1 = tf.complex(0, phi1)
    # x2 = tf.complex(0, phi2)
    # return tf.arg(tf.exp(x1) / tf.exp(x2))

def phi_bound(x):
    # assumes that two angles are in [0,2pi] before subtraction
    return tf.minimum(x, 2*np.pi - x)
    # return K.abs(tf.atan(tf.tan(x)))
    # return tf.acos(tf.cos(x))

def detaphi_map(x):
    eta = K.expand_dims(x[:,0,:], axis=1)
    phi = K.expand_dims(x[:,1,:], axis=1)
    return K.concatenate([eta, phi_bound(phi)], axis=1)

class KMeans(Layer):
    def __init__(self, k,
                 R=0.4,
                 flat_unclustered=False,
                 linear_unclustered=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 etaphi=False,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(KMeans, self).__init__(**kwargs)
        self.k = k
        self.R0 = R
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True
        self.etaphi = etaphi
        self.flat_unclustered = flat_unclustered
        self.linear_unclustered = linear_unclustered

    def build(self, input_shape):
        assert len(input_shape) == 2
        self.input_dim = input_shape[-1] # shape should be (batch_size, input_dim)

        self.centers = self.add_weight(shape=(1, self.input_dim, self.k),
                                        initializer=self.kernel_initializer,
                                        name='kernel',
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)
        self.input_spec = InputSpec(min_ndim=2, axes={-1: self.input_dim})
        self.built = True

    def call(self, inputs):
        # (batch_size, input_dim, newaxis) - (1, input_dim, k) = (batch_size, input_dim, k)
        diff = K.expand_dims(inputs) - self.centers
        if self.etaphi:
            diff = detaphi_map(diff) / self.R0
        d = K.square(diff)
        R = K.sum(d, axis=1) # sum over input_dim axis -> (batch_size, k)
        if self.flat_unclustered:
            ones = K.expand_dims(K.ones_like(inputs[:,0])) # (batch_size,1)
            out = K.concatenate([R, ones], axis=-1)
            # out = K.relu(out - 1) + 1
        elif self.linear_unclustered is not None:
            # R is the Euclidean distance in our input space
            # now compute a linear-radial distance for far-away points
            a = self.linear_unclustered; b = 1 - a # gives an intersection at R=R0
            lrd = a * K.sqrt(R) + b
            out = K.minimum(R, lrd) # assumes a <= 1, so that there is only one positive intersection
        else:
            out = R
        return out

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        assert input_shape[1]
        output_shape = list(input_shape)
        if self.flat_unclustered:
            output_shape[1] = self.k + 1
        else:
            output_shape[1] = self.k
        return tuple(output_shape)

    def get_config(self):
        config = {
            'k': self.k,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
        }
        base_config = super(KMeans, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DenseBroadcast(Layer):
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(DenseBroadcast, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[1] # always use the second dimension and broadcast over rest

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,1,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={1: input_dim})
        self.built = True

    def call(self, inputs):
        output = tf.einsum('ijk,jl->ilk', inputs, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[1]
        output_shape = list(input_shape)
        output_shape[1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(DenseBroadcast, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def _inner_product(v0, v1):
    # x0_component = v0[:,0] * v1[:,0]
    # x1_component = v0[:,1] * v1[:,1]
    # x2_component = v0[:,2] * v1[:,2]
    # t_component = v0[:,3] * v1[:,3]
    # return (x0_component + x1_component + x2_component - t_component)
    x = tf.einsum('ij,ij->i', v0[:,:3], v1[:,:3])
    t = v0[:,3] * v1[:,3]
    return x - t


class LorentzInnerCell(Layer):
    """Cell class for LorentzInner.

    # Arguments
        activation: Activation function to use
            Default: linear (`linear`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
    """
    def __init__(self,
                 activation='linear',
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(LorentzInnerCell, self).__init__(**kwargs)
        self.units = 2
        self.activation = activations.get(activation)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = (2, 4) # 4-vector
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    def build(self, input_shape):
        # nein, mein Freund!
        self.built = True

    def call(self, inputs, states, training=None):
        vec0 = states[1]
        vec1 = inputs
        N = K.shape(vec0)[0]

        ip = _inner_product(vec0, vec1)

        diff = vec0 - vec1
        d01 = _inner_product(diff, diff)

        output = self.activation(K.stack([ip, d01], axis=1))
        #output = K.reshape(self.activation(K.stack([ip, d01], axis=1)), (N, -1))
        return output, [output, vec1] # new input is state for next call

    def get_config(self):
        config = {'activation': activations.serialize(self.activation),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(LorentzInnerCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LorentzInner(RNN):
    """RNN in which each step computers the Lorentz inner product of the current
       and previous 4-vectors

    # Arguments
        activation: Activation function to use
            Default: 'linear'
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        return_sequences: Boolean. Whether to return the last output.
            in the output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state
            in addition to the output.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards and return the
            reversed sequence.
        unroll: Boolean (default False).
            If True, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.
    """

    def __init__(self,
                 activation='linear',
                 dropout=0.,
                 recurrent_dropout=0.,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 unroll=False,
                 **kwargs):

        cell = LorentzInnerCell(activation=activation,
                                dropout=dropout,
                                recurrent_dropout=recurrent_dropout)
        super(LorentzInner, self).__init__(cell,
                                           return_sequences=return_sequences,
                                           return_state=return_state,
                                           go_backwards=go_backwards,
                                           unroll=unroll,
                                           **kwargs)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        return super(LorentzInner, self).call(inputs,
                                              mask=mask,
                                              training=training,
                                              initial_state=initial_state)

    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_config(self):
        config = {'activation': activations.serialize(self.activation),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(LorentzInner, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if 'implementation' in config:
            config.pop('implementation')
        return cls(**config)


class LorentzOuterCell(Layer):
    """Cell class for LorentzOuter.

    # Arguments
        activation: Activation function to use
            (see [activations](../activations.md)).
            Default: linear (`linear`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
    """

    def __init__(self,
                 activation='linear',
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(LorentzOuterCell, self).__init__(**kwargs)
        self.units = 16
        self.activation = activations.get(activation)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = (16,4) # 4-vector
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    def build(self, input_shape):
        # nein, mein Freund!
        self.built = True

    def call(self, inputs, states, training=None):
        vec0 = states[1]
        vec1 = inputs

        N = K.shape(vec0)[0]
        op = K.reshape(vec0[:,:,np.newaxis] *
                       vec1[:,np.newaxis,:],
                       (N, -1))
        output = self.activation(op)

        return output, [output,vec1] # new input is state for next call

    def get_config(self):
        config = {'activation': activations.serialize(self.activation),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(LorentzOuterCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LorentzOuter(RNN):
    """RNN in which each step computers the Lorentz inner product of the current
       and previous 4-vectors

    # Arguments
        activation: Activation function to use
            (see [activations](../activations.md)).
            Default: 'linear'
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        return_sequences: Boolean. Whether to return the last output.
            in the output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state
            in addition to the output.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards and return the
            reversed sequence.
        unroll: Boolean (default False).
            If True, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.
    """

    def __init__(self,
                 activation='linear',
                 dropout=0.,
                 recurrent_dropout=0.,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 unroll=False,
                 **kwargs):

        cell = LorentzOuterCell(activation=activation,
                                dropout=dropout,
                                recurrent_dropout=recurrent_dropout)
        super(LorentzOuter, self).__init__(cell,
                                           return_sequences=return_sequences,
                                           return_state=return_state,
                                           go_backwards=go_backwards,
                                           unroll=unroll,
                                           **kwargs)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        return super(LorentzOuter, self).call(inputs,
                                              mask=mask,
                                              training=training,
                                              initial_state=initial_state)

    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_config(self):
        config = {'activation': activations.serialize(self.activation),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(LorentzOuter, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if 'implementation' in config:
            config.pop('implementation')
        return cls(**config)


class PolyLayer(Layer):
    def __init__(self, order, init=None, return_coeffs=False, alpha=0.01, **kwargs):
        self.return_coeffs = return_coeffs
        self.order = order
        self.output_dim = order + 1
        self.alpha = alpha
        self._init = init
        self._norm = K.constant([[1./x] for x in range(1,order+2)])
        super(PolyLayer, self).__init__(**kwargs)

    @property
    def coeffs(self):
        return K.eval(self.kernel)
    @property
    def integral(self):
        return K.eval(self._integral)
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.order + 1, 1),
                                      initializer='uniform',
                                      trainable=True)
        if self._init is not None:
            self._init = [[x] for x in self._init]
            K.set_value(self.kernel, self._init)
        super(PolyLayer, self).build(input_shape)

    def call(self, x):
        basis = K.concatenate([K.pow(x, i) for i in xrange(self.order + 1)]) # basis vector
        self._integral = K.sum(self.kernel * self._norm) # normalize the integral
        #self._norm = tf.Print(self._norm, [self._norm])
        #self._integral = tf.Print(self._integral, [self._integral])
        mask = K.abs(x - 0.5) <= 0.5 # between 0 and 1
        # likelihood
        l = tf.where(mask,
                     K.dot(basis, self.kernel) / self._integral,
                     -(self.alpha*K.abs(x-0.5) - self.alpha*0.5))
        penalty = K.zeros_like(l) + K.square(1 - self._integral) + K.epsilon()
        loss = -l + 100*penalty
        if self.return_coeffs:
            coeffs = K.transpose(K.repeat_elements(self.kernel, rep=K.shape(x)[0], axis=1))
            return K.concatenate([loss, coeffs])
        else:
            return loss

    def compute_output_shape(self, input_shape):
        if self.return_coeffs:
            return (input_shape[0], self.order + 2)
        else:
            return (input_shape[0], 1)

def choose(n, k):
    return np.math.factorial(n) / (np.math.factorial(k) * np.math.factorial(n - k))


class WeightLayer(Layer):
    def __init__(self, poly_layer, **kwargs):
        super(WeightLayer, self).__init__(**kwargs)
        self.parent = poly_layer
    def build(self, input_shape):
        super(WeightLayer, self).build(input_shape)
    def call(self, x):
        ones = K.flatten(K.ones_like(x))
        flat = K.flatten(self.parent.kernel)
        reshape = tf.einsum('i,j->ij', ones, flat)
        return K.expand_dims(reshape, axis=1)
    def compute_output_shape(self, input_shape):
        order = self.parent.order
        return (input_shape[0], 1, (order + 1) ** 2)

class ConvexPolyLayer(Layer):
    def __init__(self, order, init=None, alpha=0.0, weighted=False, **kwargs):
        super(ConvexPolyLayer, self).__init__(**kwargs)
        self.order = order
        self.output_dim = order + 1
        self.alpha = alpha
        self._init = init
        self._weighted = weighted
        vals = [[0 for _ in self.powers] for __ in self.powers]
        for a in self.powers:
            for b in self.powers:
                for k in xrange(b+1):
                    vals[a][b] += (np.power(-1, k) * choose(b, k) * 1. / (a + k + 1))
        self._norm = K.constant(vals)

    @property
    def powers(self):
        return xrange(self.order + 1)

    @property
    def coeffs(self):
        return K.eval(self.kernel)

    @property
    def poly_coeffs(self):
        base = self.coeffs
        cs = {x:0 for x in xrange(self.order * 2 + 1)}
        for a in self.powers:
            for b in self.powers:
                c = base[a, b]
                for k in xrange(b+1):
                    cs[a + k] += c * np.power(-1, k) * choose(b, k)
        return [[cs[x]] for x in xrange(self.order * 2 + 1)]

    @property
    def integral(self):
        return K.eval(self._integral)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.order + 1, self.order + 1),
                                      initializer='ones',
                                      trainable=True)
        if self._init is not None:
            self._init = [[x] for x in self._init]
            K.set_value(self.kernel, self._init)
        super(ConvexPolyLayer, self).build(input_shape)

    def call(self, x_):
        if self._weighted:
            x = x_[:,:,0]
            w = x_[:,:,1]
        else:
            x = x_
            w = K.ones_like(x)
        basis0 = K.expand_dims(K.concatenate([K.pow(x, i) for i in self.powers]))
        one = K.ones_like(x)
        basis1 = K.expand_dims(K.concatenate([K.pow(one - x, i) for i in self.powers]))
        basis = K.batch_dot(basis0, basis1, axes=2)
        self._integral = K.sum(self.kernel * self._norm) # normalize the integral
        prob = K.expand_dims(K.sum(K.sum(basis * self.kernel, axis=1), axis=1) / self._integral)
        mask = K.abs(x - 0.5) <= 0.5 # between 0 and 1
        # likelihood
        l = tf.where(mask, -K.log(prob),
                     self.alpha * (K.abs(x-0.5) - 0.5))
        l += K.sum(K.relu(-self.kernel)) / K.max(K.abs(self.kernel)) # kernel coeffs should always be positiv
        return l * w

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)



# https://github.com/michetonu/gradient_reversal_keras_tf/blob/master/flipGradientTF.py
def _reverse(x, scale = 1):
    # first make this available to tensorflow
    if hasattr(_reverse, 'N'):
        _reverse.N += 1
    else:
        _reverse.N = 1
    name = 'reverse%i'%_reverse.N

    @tf.RegisterGradient(name)
    def f(op, g):
        # this is the actual tensorflow op
        return [scale * tf.negative(g)]

    graph = K.get_session().graph
    with graph.gradient_override_map({'Identity':name}):
        ret = tf.identity(x)

    return ret

class GradReverseLayer(Layer):
    def __init__(self, scale = 1, **kwargs):
        self.scale = scale
        super(GradReverseLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.trainable_weights = []
        super(GradReverseLayer, self).build(input_shape)

    def call(self, x):
        return _reverse(x, self.scale)

    def get_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {}
        base_config = super(GradReverseLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Adversary(object):
    def __init__(self, n_output_bins, n_outputs=1, scale=1):
        self.scale = scale
        self.n_output_bins = n_output_bins
        self.n_outputs = n_outputs
        self._outputs = None
        self._dense = []

    def __call__(self, inputs):
        self._reverse = GradReverseLayer(self.scale, name='u_grl')(inputs)

        n_outputs = self.n_outputs
        self._dense.append( [Dense(5, activation='tanh')(self._reverse) for _ in xrange(n_outputs)] )
        self._dense.append( [Dense(10, activation='tanh')(d) for d in self._dense[-1]] )
        self._dense.append( [Dense(10, activation='tanh')(d) for d in self._dense[-1]] )
        self._dense.append( [Dense(10, activation='tanh')(d) for d in self._dense[-1]] )
        if self.n_output_bins > 1:
            if n_outputs == 1:
                self._outputs = [Dense(self.n_output_bins, activation='softmax', name='adv')(d)
                                 for d in self._dense[-1]]
            else:
                self._outputs = [Dense(self.n_output_bins, activation='softmax', name='adv%i'%i)(d)
                                 for i,d in enumerate(self._dense[-1])]
        else:
            if n_outputs == 1:
                self._outputs = [Dense(1, activation='linear', name='adv')(d) for d in self._dense[-1]]
            else:
                self._outputs = [Dense(1, activation='linear', name='adv%i'%i)(d) for i,d in enumerate(self._dense[-1])]
        return self._outputs
