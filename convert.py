# -*- coding: utf-8 -*-
# /usr/bin/python2


from __future__ import print_function

import numpy as np
from utils import *
from hparams import Hyperparams as hp
import tensorflow as tf
from data_load import load_data
from scipy.io.wavfile import write
from models import Graph

def convert():
    # Load data
    _lengths, _mfccs = load_data(mode="convert")

    mfccs = np.zeros((len(_lengths), max(_lengths), hp.n_mfccs), dtype=np.float32)
    for i in range(len(_mfccs)):
        mfcc = _mfccs[i]
        mfccs[i, :len(mfcc), :] = mfcc

    # Graph
    g = Graph("convert"); print("Conversion Graph loaded")
    with tf.Session() as sess:
        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Restore saved variables
        logdir = hp.logdir + "/train1"
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net1')
        saver = tf.train.Saver(var_list=var_list)
        ckpt = tf.train.latest_checkpoint(logdir)
        if ckpt is not None: saver.restore(sess, ckpt)

        logdir = hp.logdir + "/train2"
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net2')
        saver2 = tf.train.Saver(var_list=var_list)
        ckpt = tf.train.latest_checkpoint(logdir)
        if ckpt is not None: saver2.restore(sess, ckpt)

        # Evaluation
        mags = sess.run(g.mag_hats, {g.mfccs: mfccs})
        for i, (mag, l) in enumerate(zip(mags, _lengths)):
            wav = spectrogram2wav(mag[:l])
            write('50lang-output/{}.wav'.format(i+1), hp.sr, wav)

if __name__ == '__main__':
    convert(); print("Done")