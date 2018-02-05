# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/cross_vc2
'''

from __future__ import print_function

from hparams import Hyperparams as hp
from modules import *

def net1(inputs, training=True, scope="net1", reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        # Pre-net
        prenet_output = prenet(inputs,
                         num_units=[hp.hidden_units, hp.hidden_units // 2],
                         training=training)  # (N, t, H/2)
        # CBHG
        # conv1d banks
        tensor = conv1d_banks(prenet_output,
                              K=hp.num_banks,
                              activation_fn=tf.nn.relu)
        # pooling
        tensor = tf.layers.max_pooling1d(tensor, 2, 1, padding="same")  # (N, T, D)

        # conv1d projections
        tensor = conv1d(tensor,
                        filters=hp.hidden_units//2,
                        size=3,
                        activation_fn=tf.nn.relu,
                        scope="conv1d_1")  # (N, T, D)
        tensor = conv1d(tensor,
                        filters=hp.hidden_units//2,
                        size=3,
                        activation_fn=None,
                        scope="conv1d_2")  # (N, T, n_mfccs)

        tensor += prenet_output  # (N, T, n_mfccs) # residual connections

        tensor = tf.layers.dense(tensor, hp.hidden_units // 2)  # (N, T_y, E/2)

        # highwaynet
        for i in range(hp.num_highwaynet_blocks):
            tensor = highwaynet(tensor,
                             num_units=hp.hidden_units//2,
                             scope='highwaynet_{}'.format(i))  # (N, T, D)

        # bidirection grus
        tensor = gru(tensor, hp.hidden_units//2, True)  # (N, T, 2D)

        # Readout
        logits = tf.layers.dense(tensor, len(hp.vocab))
    return logits

def net2(inputs, training=True, scope="net2", reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        i = 1
        tensor = conv1d(inputs,
                        filters=hp.hidden_units,
                        size=1,
                        rate=1,
                        padding="SAME",
                        dropout_rate=hp.dropout_rate,
                        training=training,
                        scope="C_{}".format(i)); i += 1
        for j in range(4):
            tensor = hc(tensor,
                            size=3,
                            rate=3**j,
                            padding="SAME",
                            dropout_rate=hp.dropout_rate,
                            training=training,
                            scope="HC_{}".format(i)); i += 1

        for _ in range(2):
            tensor = hc(tensor,
                            size=3,
                            rate=1,
                            padding="SAME",
                            dropout_rate=hp.dropout_rate,
                            training=training,
                            scope="HC_{}".format(i)); i += 1
        for _ in range(3):
            tensor = conv1d(tensor,
                            size=1,
                            rate=1,
                            padding="SAME",
                            dropout_rate=hp.dropout_rate,
                            activation_fn=tf.nn.relu,
                            training=training,
                            scope="C_{}".format(i)); i += 1
        # mag_hats
        logits = conv1d(tensor,
                        filters=1+hp.n_fft//2,
                        size=1,
                        rate=1,
                        padding="SAME",
                        dropout_rate=hp.dropout_rate,
                        training=training,
                        scope="C_{}".format(i)); i += 1

        return logits
