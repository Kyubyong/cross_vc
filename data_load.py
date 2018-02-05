# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/cross_vc
'''

from __future__ import print_function

from hparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from utils import *
import codecs
import re
import os, glob
import tqdm



def load_vocab():
    phn2idx = {phn: idx for idx, phn in enumerate(hp.vocab)}
    idx2phn = {idx: phn for idx, phn in enumerate(hp.vocab)}
    return phn2idx, idx2phn

def load_data(mode="train1"):
    if mode in ("train1", "eval1"):
        wav_fpaths = glob.glob(hp.timit)
        # wav_fpaths = [w for w in wav_fpaths if 'TEST/DR1/FAKS' not in w]
        phn_fpaths = [f.replace("WAV.wav", "PHN").replace("wav", 'PHN')
                      for f in wav_fpaths]
        if mode=="train1":
            return wav_fpaths[hp.batch_size:], phn_fpaths[hp.batch_size:]
        else:
            wav_fpaths, phn_fpaths = wav_fpaths[:hp.batch_size], phn_fpaths[:hp.batch_size]

            mfccs = np.zeros((hp.batch_size, 1500, hp.n_mfccs), np.float32)
            phns = np.zeros((hp.batch_size, 1500), np.int32)
            max_length = 0
            for i, (w, p) in enumerate(zip(wav_fpaths, phn_fpaths)):
                mfcc, phone = load_mfccs_and_phones(w, p)
                max_length = max(max_length, len(mfcc))
                mfccs[i, :len(mfcc), :] = mfcc
                phns[i, :len(phone)] = phone
            mfccs = mfccs[:, :max_length, :]
            phns = phns[:, :max_length]

            return mfccs, phns

    elif mode=="train2":
        wav_fpaths = glob.glob(hp.arctic)
        return wav_fpaths

    else: # convert
        files = glob.glob(hp.test_data)
        mfccs = np.zeros((len(files), 800, hp.n_mfccs), np.float32)
        for i, f in enumerate(files):
            mfcc, _ = get_mfcc_and_mag(f, trim=True)
            mfcc = mfcc[:800]
            mfccs[i, :len(mfcc), :] = mfcc
        return mfccs

def load_mfccs_and_phones(wav_fpath, phn_fpath):
    phn2idx, idx2phn = load_vocab()
    mfccs, _ = get_mfcc_and_mag(wav_fpath, trim=False)
    phns = np.zeros(shape=(mfccs.shape[0],), dtype=np.int32)

    # phones
    phones = [line.strip().split()[-1] for line in open(phn_fpath, 'r')]

    triphones = []
    for a, b, c in zip(["0"] + phones[:-1], phones, phones[1:] + ["0"]):
        triphones.append((a, b, c))

    for i, line in enumerate(open(phn_fpath, 'r')):
        start_point, _, phn = line.strip().split()
        bnd = int(start_point) // int(hp.sr * hp.frame_shift)  # the ordering number of frames

        triphone = triphones[i]
        if triphone in phn2idx:
            phn = phn2idx[triphone]
        elif phn in phn2idx:
            phn = phn2idx[phn]
        else: # error
            print(phn)
        phns[bnd:] = phn
    return mfccs, phns

def get_batch(mode="train1"):
    '''Loads data and put them in mini batch queues.
    mode: A string. Either `train1` or `train2`.
    '''
    # with tf.device('/cpu:0'):
    # Load data
    if mode=='train1':
        wav_fpaths, phn_fpaths = load_data(mode=mode)

        # calc total batch count
        num_batch = len(wav_fpaths) // hp.batch_size

        # Create Queues
        wav_fpath, phn_fpath = tf.train.slice_input_producer([wav_fpaths, phn_fpaths], shuffle=True)

        # Decoding
        mfccs, phones = tf.py_func(load_mfccs_and_phones, [wav_fpath, phn_fpath], [tf.float32, tf.int32])  # (T, n_mfccs)

        # Create batch queues
        mfccs, phones, wav_fpaths = tf.train.batch([mfccs, phones, wav_fpath],
                                       batch_size=hp.batch_size,
                                       num_threads=32,
                                       shapes=[(None, hp.n_mfccs), (None,), ()],
                                       dynamic_pad=True)

        return mfccs, phones, num_batch, wav_fpaths

    elif mode=='train2':
        wav_fpaths = load_data(mode=mode)
        # maxlen, minlen = max(lengths), min(lengths)

        # calc total batch count
        num_batch = len(wav_fpaths) // hp.batch_size

        # Create Queues
        wav_fpath, = tf.train.slice_input_producer([wav_fpaths])

        # Decoding
        mfcc, mag = tf.py_func(get_mfcc_and_mag, [wav_fpath], [tf.float32, tf.float32])  # (T, n_mfccs)

        # Cropping
        mfcc = mfcc[:hp.maxlen]
        mag = mag[:hp.maxlen]

        # Create batch queues
        mfccs, mags = tf.train.batch([mfcc, mag],
                                       batch_size=hp.batch_size,
                                       num_threads=32,
                                       shapes=[(None, hp.n_mfccs), (None, 1+hp.n_fft//2)],
                                       dynamic_pad=True)

        return mfccs, mags, num_batch