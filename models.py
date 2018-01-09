# -*- coding: utf-8 -*-
#!/usr/bin/env python

import os, sys

import tensorflow as tf

from utils import load_vocab
from data_load import get_batch_queue
from modules import *
from hparams import Hyperparams as hp
from tqdm import tqdm

class Graph:
    def __init__(self, mode="train1"):
        # Load vocabulary
        self.phn2idx, self.idx2phn = load_vocab()

        # Phase
        training = True if "train" in mode else False

        # Data feed
        if mode=="train1":
            self.mfccs, self.phones, self.num_batch = get_batch_queue(mode=mode)
        elif mode=="eval1":
            self.mfccs = tf.placeholder(tf.float32, shape=(None, None, hp.n_mfccs))
            self.phones = tf.placeholder(tf.int32, shape=(None, None,))
        elif mode=="train2":
            self.mfccs, self.mels, self.mags, self.num_batch = get_batch_queue(mode=mode)
        elif mode=="eval2":
            self.mfccs = tf.placeholder(tf.float32, shape=(None, None, hp.n_mfccs))
            # self.mels = tf.placeholder(tf.int32, shape=(None, None, hp.n_mels))
            self.mags = tf.placeholder(tf.float32, shape=(None, None, 1+hp.n_fft//2))
        else: # convert
            self.mfccs = tf.placeholder(tf.float32, shape=(None, None, hp.n_mfccs))

        # Graph
        ## Masks
        self.masks = tf.sign(tf.abs(self.mfccs[:,:,:1]))  # (N, t, 1)

        with tf.variable_scope("net1"):
            # Pre-net
            self.tensor = prenet(self.mfccs,
                                num_units=[hp.hidden_units, hp.hidden_units // 2],
                                dropout_rate=hp.dropout_rate,
                                training=training)  # (N, t, H/2)
            # CBHG
            self.tensor = cbhg(self.tensor,
                                  hp.num_banks,
                                  hp.hidden_units // 2,
                                  hp.num_highwaynet_blocks,
                                  hp.norm_type,
                                  False,
                                  training,
                                  tf.nn.relu)

            # Final linear projection
            def apply_masks(logits, masks):
                rev_masks = tf.to_float(tf.equal(masks, 0))
                rev_masks = rev_masks * 100000 + 1
                ones = tf.ones_like(logits[:, :, 1:])
                rev_masks = tf.concat((rev_masks, ones), -1)
                return logits * rev_masks

            self.ppg_logits = tf.layers.dense(self.tensor, len(self.phn2idx))  # (N, t, V)
            self.ppg_logits = apply_masks(self.ppg_logits, self.masks)
            self.ppgs = tf.nn.softmax(self.ppg_logits)  # (N, t, V)
            self.phone_hats = tf.to_int32(tf.argmax(self.ppgs, -1))  # (N, t)

        with tf.variable_scope("net2"):
            with tf.variable_scope("mags"):
                # Pre-net
                self.tensor = prenet(self.ppgs,
                                     num_units=[hp.hidden_units, hp.hidden_units // 2],
                                     dropout_rate=hp.dropout_rate,
                                     training=training)  # (N, T, E/2)

                # CBHG
                self.tensor = cbhg(self.tensor,
                                   hp.num_banks,
                                   hp.hidden_units // 2,
                                   hp.num_highwaynet_blocks,
                                   "bn",
                                   False,
                                   training,
                                   tf.nn.relu)

                self.mag_logits = tf.layers.dense(self.tensor, 1 + hp.n_fft // 2)  # (N, t, n_mels)
                self.mag_hats = tf.nn.sigmoid(self.mag_logits)  # (N, t, n_mels)
                # self.mag_logits = tf.expand_dims(self.mag_logits, -1) # (N, t, n_mels, 1)
                # self.mag_logits = tf.layers.dense(self.mag_logits, 100) # (N, t, n_mels, 100)
                # self.mel_hats = tf.nn.sigmoid(self.mel_logits) # (N, T, n_mels)
                # self.masks  = tf.expand_dims(self.masks, -1)
                self.mag_hats *= self.masks
            # with tf.variable_scope("mags"):
            #     # Pre-net
            #     self.tensor = prenet(self.mel_hats,
            #                          num_units=[hp.hidden_units, hp.hidden_units // 2],
            #                          dropout_rate=hp.dropout_rate,
            #                          training=training)  # (N, T, E/2)
            #
            #     # CBHG
            #     self.tensor = cbhg(self.tensor,
            #                        hp.num_banks,
            #                        hp.hidden_units // 2,
            #                        hp.num_highwaynet_blocks,
            #                        hp.norm_type,
            #                        training,
            #                        tf.nn.relu)
            #
            #     self.mag_logits = tf.layers.dense(self.tensor, 1 + hp.n_fft // 2)  # (N, t, 1+fft/2)
            #     self.mag_hats = tf.nn.sigmoid(self.mag_logits)  # (N, T, n_fft/2+1)
            #     self.mag_hats *= self.masks

        if mode in ["train1", "eval1"]:
            # Training Scheme
            with tf.variable_scope("training"):
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr)

            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.ppg_logits, labels=self.phones)
            self.loss *= tf.squeeze(self.masks, -1)
            self.loss = tf.reduce_sum(self.loss)/(tf.reduce_sum(self.masks)*len(self.phn2idx))
            tf.summary.scalar('net1/{}/loss_ppgs'.format(mode), self.loss)

            # eval
            self.hits = tf.to_float(tf.equal(self.phone_hats, self.phones))
            self.hits *= tf.squeeze(self.masks, -1)
            self.acc = tf.reduce_sum(self.hits)/tf.reduce_sum(self.masks)
            tf.summary.scalar('net1/{}/acc'.format(mode), self.acc)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net1')
                self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step, var_list=self.var_list)

        elif mode in ["train2", "eval2"]:
            # Training Scheme
            with tf.variable_scope("training2"):
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr)

            # self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.mag_logits, labels=self.mags)
            # self._loss = self.loss
            # self.loss *= self.masks
            # self.max_loss = tf.reduce_max(self.loss)
            # self.loss = tf.reduce_sum(self.loss) / tf.reduce_sum(self.masks * (1+hp.n_fft//2))
            #
            # # eval
            # self.mag_hats = tf.to_int32(tf.argmax(self.mag_logits, -1))
            # self.hits = tf.to_float(tf.equal(self.mag_hats, self.mags))
            # self.hits *= self.masks
            # self.acc = tf.reduce_sum(self.hits) / tf.reduce_sum(self.masks * (hp.n_fft//2+1))
            # tf.summary.scalar('net2/{}/acc'.format(mode), self.acc)

            # self.mel_loss = tf.square(self.mel_hats - self.mels)
            # self.max_mel_loss = tf.reduce_max(self.mel_loss)
            # self.mel_loss = tf.reduce_sum(self.mel_loss) / (tf.reduce_sum(self.masks) * (hp.n_mels))

            self.mag_loss = tf.square(self.mag_hats - self.mags)
            self.max_mag_loss = tf.reduce_max(self.mag_loss)
            self.mag_loss = tf.reduce_sum(self.mag_loss) / (tf.reduce_sum(self.masks)*(1+hp.n_fft//2))

            # self.loss = self.mel_loss + self.mag_loss

            # tf.summary.scalar('net2/{}/mel_loss'.format(mode), self.mel_loss)
            # tf.summary.scalar('net2/{}/max_mel_loss'.format(mode), self.max_mel_loss)
            # tf.summary.scalar('net2/{}/mag_loss'.format(mode), self.mag_loss)
            tf.summary.scalar('net2/{}/max_mag_loss'.format(mode), self.max_mag_loss)
            tf.summary.scalar('net2/{}/loss'.format(mode), self.mag_loss)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net2')
                self.train_op = self.optimizer.minimize(self.mag_loss, global_step=self.global_step, var_list=self.var_list)
        self.merged = tf.summary.merge_all()
