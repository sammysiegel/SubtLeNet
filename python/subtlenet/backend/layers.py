from __future__ import division

import numpy as np
import keras.backend as K
from keras.engine import Layer, InputSpec
from keras.layers import RNN
from keras import activations, initializers, regularizers, constraints
import tensorflow as tf


class LorentzInnerCell(Layer):
    """Cell class for LorentzInner.

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
        super(LorentzInnerCell, self).__init__(**kwargs)
        self.units = 1 
        self.activation = activations.get(activation)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = (1, 4) # 4-vector 
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    def build(self, input_shape):
        # nein, mein Freund!
        self.built = True

    def call(self, inputs, states, training=None):
        vec0 = states[1]
        vec1 = inputs
        N = K.shape(vec0)[0]
        
        #p_component = tf.diag(K.dot(vec0[:,:3], K.transpose(vec1[:,:3])))
        # above line allocates too much memory on GPU...hardcode the inner 
        # product for now?
        p0_component = vec0[:,0] * vec1[:,0]
        p1_component = vec0[:,1] * vec1[:,1]
        p2_component = vec0[:,2] * vec1[:,2]
        e_component = vec0[:,3] * vec1[:,3]
        
        ip = p0_component + p1_component + p2_component - e_component 
        output = K.reshape(self.activation(ip), (N, -1))

        return output, [output, vec1] # new input is state for next call

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(LorentzInnerCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class LorentzInner(RNN):
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

        cell = LorentzInnerCell(activation=activation,
                                dropout=dropout,
                                recurrent_dropout=recurrent_dropout)
        super(LorentzInner, self).__init__(cell,
                                           return_sequences=return_sequences,
                                           return_state=return_state,
                                           go_backwards=go_backwards,
                                           stateful=False,
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
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
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
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
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
                                           stateful=False,
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
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
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

