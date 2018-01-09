# -*- coding: utf-8 -*-
# /usr/bin/python2

from __future__ import print_function
from hparams import Hyperparams as hp
from tqdm import tqdm
from models import Graph
import tensorflow as tf
from data_load import load_data
import numpy as np
from utils import *
from scipy.io.wavfile import write

def train2():
    g = Graph("train2"); print("Training Graph loaded")

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

        # Writer & Queue
        writer = tf.summary.FileWriter(logdir, sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        while 1:
            for _ in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                gs, _ = sess.run([g.global_step, g.train_op])

                # Write checkpoint files regularly
                # phone_hats = sess.run(g.phone_hats)
                # print(" ".join(g.idx2phn[idx] for idx in phone_hats[0]))
                if gs % 1000 == 0:
                    merged = sess.run(g.merged)
                    writer.add_summary(merged, global_step=gs)

                    mag_hats, mags = sess.run([g.mag_hats, g.mags])
                    print("mag_hats=", mag_hats[0, :30, 0])
                    print("mags=", mags[0, :30, 0])

                    # evaluation
                    with tf.Graph().as_default():
                        eval2()

                    saver2.save(sess, logdir + '/model_gs_{}'.format(gs))

        writer.close()
        coord.request_stop()
        coord.join(threads)

def eval2():
    def dicretize(mag, num_bins=100):
        bins = np.logspace(-1, 0, num_bins - 1, base=10)
        mag = np.digitize(mag, bins).astype(np.int32)
        return mag

    # Load data
    phn2idx, idx2phn = load_vocab()
    _lengths, _mfccs, _mels, _mags = load_data(mode="eval2")

    mfccs = np.zeros((len(_lengths), max(_lengths), hp.n_mfccs), dtype=np.float32)
    mels = np.zeros((len(_lengths), max(_lengths), hp.n_mels), dtype=np.float32)
    mags = np.zeros((len(_lengths), max(_lengths), 1 + hp.n_fft // 2), dtype=np.int32)
    for i in range(len(_mfccs)):
        mfcc = np.load(_mfccs[i])
        mel = np.load(_mels[i])
        mag = np.load(_mags[i])
        # mag = dicretize(mag)
        mfccs[i, :len(mfcc), :] = mfcc
        mels[i, :len(mel), :] = mel
        mags[i, :len(mag), :] = mag


    # mels =
    # Graph
    g = Graph("eval2"); print("Evaluation Graph loaded")
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
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net2') +\
                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'training2')
        saver2 = tf.train.Saver(var_list=var_list)
        ckpt = tf.train.latest_checkpoint(logdir)
        if ckpt is not None: saver2.restore(sess, ckpt)

        # Writer
        writer = tf.summary.FileWriter(logdir, sess.graph)

        # Evaluation
        # merged, gs, mels, mel_hats, mag_hats, mags = sess.run([g.merged, g.global_step, g.mels, g.mel_hats, g.mag_hats, g.mags], {g.mfccs: mfccs, g.mels: mels, g.mags: mags})
        merged, gs, mag_hats, mags, phone_hats = sess.run([g.merged, g.global_step, g.mag_hats, g.mags, g.phone_hats], {g.mfccs: mfccs, g.mags: mags})
        # mag_hats = np.argmax(mag_logits, -1).astype(np.int32)

        # print(" ".join(g.idx2phn[idx] for idx in phone_hats[0]))
        # print("mag_logits=", mag_logits[0, ])
        # # monitor
        # mel_hat = np.argmax(mel_logits, -1)[0]
        # bins = np.logspace(-1, 0, 99, base=10)
        # bins_ = np.append(bins, bins[-1])
        # mag_hat = bins_[mag_hats[0]]
        #
        # print("mag_hat:", mag_hats[0, 10, :10])
        # print("mag    :", mags[0, 10, :10])
        #
        # np.set_printoptions(threshold=np.nan)
        # print(mag_hats[0, :100, :100])
        # print("mag:\n")
        # print(mags[0, :100, :100])
        # np.save('mel_gt.npy', mels[0])
        # np.save('mel_hat.npy', mel_hat)
        # np.save('mel_gt.npy', mels[0])
        # np.save('mag_hat.npy', mag_hats[0])
        wav = spectrogram2wav(mag_hats[0])
        wav = np.expand_dims(wav, 0)

        # Summary
        audio_summary = tf.summary.audio("audio sample", wav, hp.sr, max_outputs=1)

        # Write summaries
        writer.add_summary(merged, global_step=gs)
        writer.add_summary(sess.run(audio_summary), global_step=gs)
        writer.close()

if __name__ == '__main__':
    train2(); print("Done")
