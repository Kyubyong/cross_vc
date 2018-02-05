# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/cross_vc
'''

from __future__ import print_function

from utils import *
from hparams import Hyperparams as hp
import tensorflow as tf
from data_load import load_data
from scipy.io.wavfile import write
from graph import Graph
import os

def convert():
    g = Graph("convert"); print("Training Graph loaded")
    mfccs = load_data("convert")

    with tf.Session() as sess:
        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        # Restore
        logdir = hp.logdir + "/train1"
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net1')
        saver = tf.train.Saver(var_list=var_list)
        ckpt = tf.train.latest_checkpoint(logdir)
        if ckpt is not None: saver.restore(sess, ckpt)

        logdir = hp.logdir + "/train2"
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net2') +\
                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'training')
        saver2 = tf.train.Saver(var_list=var_list)
        ckpt = tf.train.latest_checkpoint(logdir)
        if ckpt is not None: saver2.restore(sess, ckpt)

        # Synthesize
        if not os.path.exists('50lang-output'): os.mkdir('50lang-output')

        mag_hats = sess.run(g.mag_hats, {g.mfccs: mfccs})
        for i, mag_hat in enumerate(mag_hats):
            wav = spectrogram2wav(mag_hat)
            write('50lang-output/{}.wav'.format(i+1), hp.sr, wav)

if __name__ == '__main__':
    convert(); print("Done")