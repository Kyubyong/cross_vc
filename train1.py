# -*- coding: utf-8 -*-
# /usr/bin/python2

from __future__ import print_function
from hparams import Hyperparams as hp
from tqdm import tqdm
from models import Graph
import tensorflow as tf
from data_load import load_data
import numpy as np
from utils import load_vocab

def train1():
    g = Graph(); print("Training Graph loaded")
    logdir = hp.logdir + "/train1"
    with tf.Session() as sess:
        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Restore saved variables
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net1') + \
                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'training')
        saver = tf.train.Saver(var_list=var_list)

        ckpt = tf.train.latest_checkpoint(logdir)
        if ckpt is not None: saver.restore(sess, ckpt)

        # Writer & Queue
        writer = tf.summary.FileWriter(logdir, sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Training
        while 1:
            for _ in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                gs, _ = sess.run([g.global_step, g.train_op])

            # Write checkpoint files regularly
            merged = sess.run(g.merged)
            writer.add_summary(merged, global_step=gs)

            # evaluation
            with tf.Graph().as_default():
                eval1()

            saver.save(sess, logdir + '/model_gs_{}'.format(gs))

            if gs > 15000: break

        writer.close()
        coord.request_stop()
        coord.join(threads)

def eval1():
    phn2idx, idx2phn = load_vocab()
    # Load data
    _lengths, _mfccs, _phones = load_data(mode="eval1")

    mfccs = np.zeros((len(_lengths), max(_lengths), hp.n_mfccs), dtype=np.float32)
    phones = np.zeros((len(_lengths), max(_lengths)), dtype=np.int32)
    for i in range(len(_mfccs)):
        mfcc = np.load(_mfccs[i])
        phone = np.load(_phones[i])
        mfccs[i, :len(mfcc), :] = mfcc
        phones[i, :len(phone)] = phone

    # Graph
    g = Graph("eval1"); print("Evaluation Graph loaded")
    logdir = hp.logdir + "/train1"

    with tf.Session() as sess:
        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Restore saved variables
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net1') +\
                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'training')
        saver = tf.train.Saver(var_list=var_list)

        ckpt = tf.train.latest_checkpoint(logdir)
        if ckpt is not None: saver.restore(sess, ckpt)

        # Writer
        writer = tf.summary.FileWriter(logdir, sess.graph)

        # Evaluation
        merged, acc, gs, phones, phone_hat = sess.run([g.merged, g.acc, g.global_step, g.phones, g.phone_hats], {g.mfccs: mfccs, g.phones: phones})

        # logging
        print("gt:", " ".join(idx2phn[idx] for idx in phones[-1, :]))
        print("hat:", " ".join(idx2phn[idx] for idx in phone_hat[-1, :]))
        #  Write summaries
        writer.add_summary(merged, global_step=gs)
        writer.close()

    return acc

if __name__ == '__main__':
    train1(); print("Done")
