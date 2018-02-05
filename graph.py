# -*- coding: utf-8 -*-
#!/usr/bin/env python
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/cross_vc
'''

import os, sys

import tensorflow as tf

from data_load import get_batch, load_vocab
from modules import *
from hparams import Hyperparams as hp
from tqdm import tqdm
from utils import *
from networks import net1, net2

class Graph:
    def __init__(self, mode="train1"):
        # Load vocabulary
        self.phn2idx, self.idx2phn = load_vocab()

        # Set flag
        training = True if "train" in mode else False

        # Data feed
        ## mfccs: (N, t, n_mfccs)
        if mode=="train1":
            self.mfccs, self.phones, self.num_batch, self.fname = get_batch(mode=mode)
        elif mode=="eval1":
            self.mfccs = tf.placeholder(tf.float32, shape=(None, None, hp.n_mfccs))
            self.phones = tf.placeholder(tf.int32, shape=(None, None,))
        elif mode=="train2":
            self.mfccs, self.mags, self.num_batch = get_batch(mode=mode)
        else: # convert
            self.mfccs = tf.placeholder(tf.float32, shape=(None, None, hp.n_mfccs))

        # Net1
        self.ppg_logits = net1(self.mfccs, training=training, scope="net1")
        self.ppgs = tf.nn.softmax(self.ppg_logits)  # (N, t, V)
        self.phone_hats = tf.to_int32(tf.argmax(self.ppgs, -1))  # (N, t)

        # Net2
        self.mag_logits = net2(self.ppgs, training=training, scope="net2")
        self.mag_hats = tf.nn.sigmoid(self.mag_logits)  # (N, t, V)

        self.audio = tf.py_func(spectrogram2wav, [self.mag_hats[0]], tf.float32)

        # Training Scheme
        if mode in ["train1", "eval1"]:
            with tf.variable_scope("training"):
                self.global_step = tf.Variable(0, name='global_step', trainable=False)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr)

            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.ppg_logits,
                                                                       labels=self.phones)
            self.loss = tf.reduce_mean(self.loss)

            # eval
            self.hits = tf.to_float(tf.equal(self.phone_hats, self.phones))
            self.acc = tf.reduce_mean(self.hits)

            tf.summary.scalar('net1/{}/acc'.format(mode), self.acc)

            # summary
            tf.summary.image("net1/{}/ppgs_hat".format(mode),
                             tf.expand_dims(tf.transpose(self.ppgs, [0, 2, 1]), -1),
                             max_outputs=1)
            tf.summary.image("net1/{}/phones_gt".format(mode),
                             tf.expand_dims(tf.transpose(tf.one_hot(self.phones, len(self.phn2idx)), [0, 2, 1]),-1),
                             max_outputs=1)
            tf.summary.scalar('net1/{}/loss_ppgs'.format(mode), self.loss)
            tf.summary.scalar('net1/{}/acc'.format(mode), self.acc)

            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net1')
            self.train_op = self.optimizer.minimize(self.loss,
                                                    global_step=self.global_step,
                                                    var_list=self.var_list)
        elif mode=="train2":
            # Training Scheme
            with tf.variable_scope("training2"):
                self.global_step = tf.Variable(0, name='global_step', trainable=False)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr)

            self.loss = tf.abs(self.mag_hats - self.mags)  # (N, T, n_fft//2+1)
            self.loss = tf.reduce_mean(self.loss)

            tf.summary.scalar('net2/{}/loss'.format(mode), self.loss)
            tf.summary.image('net2/{}/mag_gt'.format(mode),
                             tf.expand_dims(self.mags, -1),
                             max_outputs=1)
            tf.summary.image('net2/{}/mag_hat'.format(mode),
                             tf.expand_dims(self.mag_hats, -1),
                             max_outputs=1)
            tf.summary.audio('net2/{}/wavs'.format(mode), tf.expand_dims(self.audio, 0), hp.sr,
                             max_outputs=1)

            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net2')
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step,
                                                    var_list=self.var_list)


        self.merged = tf.summary.merge_all()

