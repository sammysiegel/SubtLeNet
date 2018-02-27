from __future__ import division

import numpy as np
import keras.backend as K
from keras.engine import Layer, InputSpec
from keras.layers import RNN, Dense
from keras import activations, initializers, regularizers, constraints
import tensorflow as tf
from .. import config


class DenseBroadcast(Layer):
    """
    `DenseBroadcast` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`). It is practically identical
    to `Dense`, except if `input` has rank greater than 2, it is not flattened,
    but rather the dot is broadcast across the remaining dimensions.

    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
        bias_initializer: Initializer for the bias vector
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
        bias_regularizer: Regularizer function applied to the bias vector
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
        bias_constraint: Constraint function applied to the bias vector

    # Input shape
        nD tensor with shape: `(batch_size, input_dim, ...)`.

    # Output shape
        nD tensor with shape: `(batch_size, units, ...)`.
    """

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


# polynomial layer - currently unused
class PolyLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.order = output_dim
        self.output_dim = output_dim
        super(PolyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(self.order + 1, 1),
                                      initializer='uniform',
                                      trainable=True)
        super(PolyLayer, self).build(input_shape)  

    def call(self, x):
        basis = K.concatenate([K.pow(x, i) for i in xrange(self.order + 1)])
        return K.dot(basis, self.kernel)

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
        self._reverse = GradReverseLayer(self.scale)(inputs)

        n_outputs = self.n_outputs
        self._dense.append( [Dense(5, activation='relu')(self._reverse) for _ in xrange(n_outputs)] )
        self._dense.append( [Dense(10, activation='relu')(d) for d in self._dense[-1]] )
        self._dense.append( [Dense(10, activation='relu')(d) for d in self._dense[-1]] )
        self._dense.append( [Dense(10, activation='tanh')(d) for d in self._dense[-1]] )
        if config.bin_decorr:
            self._outputs = [Dense(self.n_output_bins, activation='softmax')(d) for d in self._dense[-1]]
        else:
            self._outputs = [Dense(1, activation='linear')(d) for d in self._dense[-1]]
        return self._outputs 
