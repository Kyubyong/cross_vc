# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import print_function

import tensorflow as tf

def normalize(inputs,
              type="bn",
              training=True,
              activation_fn=None,
              reuse=None,
              scope="normalize"):
    '''Applies {batch|layer} normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`. If type is `bn`, the normalization is over all but
        the last dimension.
      type: A string. Either "bn" or "ln".
      is_training: Whether or not the layer is in training mode. W
      activation_fn: Activation function.
      scope: Optional scope for `variable_scope`.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    if type == "bn":
        outputs = tf.layers.batch_normalization(inputs=inputs,
                                                training=training,
                                                reuse=reuse)
    elif type == "ln":
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) * tf.rsqrt(variance + 1e-8)
            outputs = gamma * normalized + beta
    else:
        outputs = inputs

    if activation_fn is not None:
        outputs = activation_fn(outputs)

    return outputs


def conv1d(inputs,
           filters=None,
           size=1,
           rate=1,
           padding="SAME",
           dropout_rate=0,
           use_bias=True,
           norm_type=None,
           activation_fn=None,
           training=True,
           scope="conv1d",
           reuse=None):
    '''
    Args:
      inputs: A 3-D tensor with shape of [batch, time, depth].
      filters: An int. Number of outputs (=activation maps)
      size: An int. Filter size.
      rate: An int. Dilation rate.
      padding: Either `same` or `valid` or `causal` (case-insensitive).
      use_bias: A boolean.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A masked tensor of the same shape and dtypes as `inputs`.
    '''
    with tf.variable_scope(scope):
        if padding.lower() == "causal":
            # pre-padding for causality
            pad_len = (size - 1) * rate  # padding size
            inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
            padding = "valid"

        if filters is None:
            filters = inputs.get_shape().as_list()[-1]

        params = {"inputs": inputs, "filters": filters, "kernel_size": size,
                  "dilation_rate": rate, "padding": padding, "use_bias": use_bias, "reuse": reuse}

        tensor = tf.layers.conv1d(**params)
        tensor = normalize(tensor, type=norm_type, training=training)
        if activation_fn is not None:
            tensor = activation_fn(tensor)

        tensor = tf.layers.dropout(tensor, rate=dropout_rate, training=training)

    return tensor

def conv1d_banks(inputs,
                 K=16,
                 num_units=None,
                 norm_type=None,
                 use_bias=True,
                 training=True,
                 activation_fn=None,
                 scope="conv1d_banks",
                 reuse=None):
    '''Applies a series of conv1d separately.

    Args:
      inputs: A 3d tensor with shape of [N, T, C]
      K: An int. The size of conv1d banks. That is,
        The `inputs` are convolved with K filters: 1, 2, ..., K.
      is_training: A boolean. This is passed to an argument of `batch_normalize`.

    Returns:
      A 3d tensor with shape of [N, T, K*Hp.embed_size//2].
    '''
    if num_units is None:
        num_units = inputs.get_shape().as_list()[-1]

    with tf.variable_scope(scope, reuse=reuse):
        outputs = []
        for k in range(1, K + 1):
            with tf.variable_scope("num_{}".format(k)):
                output = conv1d(inputs,
                                filters=num_units,
                                size=k,
                                use_bias=use_bias,
                                norm_type=norm_type,
                                training=training,
                                activation_fn=activation_fn)
                outputs.append(output)
    outputs = tf.concat(outputs, -1)
    return outputs  # (N, T, Hp.embed_size//2*K)


def gru(inputs, num_units=None, bidirection=False, scope="gru", reuse=None):
    '''Applies a GRU.
    
    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: An int. The number of hidden units.
      bidirection: A boolean. If True, bidirectional results 
        are concatenated.
      scope: Optional scope for `variable_scope`.  
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns:
      If bidirection is True, a 3d tensor with shape of [N, T, 2*num_units],
        otherwise [N, T, num_units].
    '''
    if num_units is None:
        num_units = inputs.get_shape().as_list[-1]

    with tf.variable_scope(scope, reuse=reuse):
        cell = tf.contrib.rnn.GRUCell(num_units)
        if bidirection: 
            cell_bw = tf.contrib.rnn.GRUCell(num_units)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell_bw, inputs, dtype=tf.float32)
            return tf.concat(outputs, 2)  
        else:
            outputs, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
            return outputs

def prenet(inputs, num_units=None, dropout_rate=0., training=True, activation_fn=tf.nn.relu, scope="prenet", reuse=None):
    '''Prenet for Encoder and Decoder.
    Args:
      inputs: A 3D tensor of shape [N, T, hp.embed_size].
      is_training: A boolean.
      scope: Optional scope for `variable_scope`.  
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns:
      A 3D tensor of shape [N, T, num_units/2].
    '''
    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.layers.dense(inputs, units=num_units[0], activation=activation_fn, name="dense1")
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training, name="dropout1")
        outputs = tf.layers.dense(outputs, units=num_units[1], activation=activation_fn, name="dense2")
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training, name="dropout2")

    return outputs # (N, T, num_units/2)

def highwaynet(inputs, num_units=None, scope="highwaynet", reuse=None):
    '''Highway networks, see https://arxiv.org/abs/1505.00387

    Args:
      inputs: A 3D tensor of shape [N, T, W].
      num_units: An int or `None`. Specifies the number of units in the highway layer
             or uses the input size if `None`.
      scope: Optional scope for `variable_scope`.  
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3D tensor of shape [N, T, W].
    '''
    if not num_units:
        num_units = inputs.get_shape()[-1]
        
    with tf.variable_scope(scope, reuse=reuse):
        H = tf.layers.dense(inputs, units=num_units, activation=tf.nn.relu, name="dense1")
        T = tf.layers.dense(inputs, units=num_units, activation=tf.nn.sigmoid, bias_initializer=tf.constant_initializer(-1.0), name="dense2")
        outputs = H * T + inputs * (1. - T)
    return outputs


def cbhg(input,
         num_banks,
         num_units,
         num_highway_blocks,
         norm_type=None,
         use_bias=True,
         training=True,
         activation_fn=None,
         scope="cbhg",
         reuse=False):

    with tf.variable_scope(scope, reuse=reuse):
        # conv1d banks
        tensor = conv1d_banks(input,
                           K=num_banks,
                           num_units=num_units,
                           norm_type=norm_type,
                           use_bias=use_bias,
                           training=training,
                           activation_fn=activation_fn)  # (N, T, D)
        # pooling
        tensor = tf.layers.max_pooling1d(tensor, 2, 1, padding="same")  # (N, T, D)

        # conv1d projections
        for i in range(2):
            tensor = conv1d(tensor,
                            filters=num_units,
                            size=3,
                            norm_type=norm_type,
                            use_bias=use_bias,
                            training=training,
                            activation_fn=activation_fn if i==0 else None,
                            scope="conv1d_1_{}".format(i))  # (N, T, D)
        tensor += input  # (N, T, D) # residual connections

        # highwaynet
        for i in range(num_highway_blocks):
            tensor = highwaynet(tensor,
                             num_units=num_units,
                             scope='highwaynet_{}'.format(i))  # (N, T, D)

        # bidirection grus
        tensor = gru(tensor, num_units, True)  # (N, T, 2D)

    return tensor