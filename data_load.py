# -*- coding: utf-8 -*-
# /usr/bin/python2

import glob

import tensorflow as tf

from hparams import Hyperparams as hp
from utils import *


def load_data(mode):
    if mode in ("train1", "eval1"):
        lengths, mfccs, phones = [], [], [] # file paths
        files = glob.glob('nick/phns/*.npy') + glob.glob('timit/phns/*.npy')
        if mode=="train1":
            files = files[hp.batch_size:]
        else:
            files = files[:hp.batch_size]

        for f in files:
            lengths.append(len(np.load(f))) # frame length T
            phones.append(f)
            mfccs.append(f.replace('phns', 'mfccs'))
        return lengths, mfccs, phones # file paths

    elif mode in ("train2", "eval2"):
        lengths, mfccs, mels, mags = [], [], [], [] # file paths
        files = glob.glob('{}/mfccs/*.npy'.format(hp.data))
        if mode=="train2":
            files = files[hp.batch_size:]
        else:
            files = files[:hp.batch_size]
            print("eval files=", files)

        for f in files:
            length = len(np.load(f))
            # print(length)
            if length>800:
                if mode=="eval2":
                    print "skipped {}".format(f)
                continue
            lengths.append(length)
            mfccs.append(f)
            mels.append(f.replace('mfccs', 'mels'))
            mags.append(f.replace('mfccs', 'mags'))
        print(len(lengths))
        return lengths, mfccs, mels, mags
    else: # convert -> numpy arrays
        lengths, mfccs = [], []
        files = glob.glob('50lang/*.wav')
        for f in files:
            mfcc, _, _ = get_audio_features(f)
            lengths.append(len(mfcc))
            mfccs.append(mfcc)

        return lengths, mfccs

def get_batch_queue(mode):
    '''Loads data and put them in mini batch queues.
    mode: A string. Either `train1` or `train2`.
    '''
    with tf.device('/cpu:0'):
        # Load data
        if mode=='train1':
            lengths, mfccs, phones = load_data(mode=mode)
            maxlen, minlen = max(lengths), min(lengths)

            # calc total batch count
            num_batch = len(mfccs) // hp.batch_size

            # Convert to tensor
            lengths = tf.convert_to_tensor(lengths)
            mfccs = tf.convert_to_tensor(mfccs)
            phones = tf.convert_to_tensor(phones)

            # Create Queues
            length, mfcc, phone = tf.train.slice_input_producer([lengths, mfccs, phones], shuffle=True)

            # Decoding
            mfcc = tf.py_func(lambda x: np.load(x), [mfcc], tf.float32)  # (T, n_mfccs)
            phone = tf.py_func(lambda x: np.load(x), [phone], tf.int32)  # (T,)

            # Set shapes
            mfcc.set_shape([None, hp.n_mfccs])
            phone.set_shape([None,])

            # Create batch queues
            _, (mfccs, phones) = tf.contrib.training.bucket_by_sequence_length(
                                                input_length=length,
                                                tensors=[mfcc, phone],
                                                batch_size=hp.batch_size,
                                                bucket_boundaries=[i for i in range(minlen+1, maxlen-1, 50)],
                                                num_threads=16,
                                                capacity=hp.batch_size * 4,
                                                dynamic_pad=True)

            return mfccs, phones, num_batch

        elif mode=='train2':
            lengths, mfccs, mels, mags = load_data(mode=mode)
            maxlen, minlen = max(lengths), min(lengths)

            # calc total batch count
            num_batch = len(mfccs) // hp.batch_size

            # Convert to tensor
            lengths = tf.convert_to_tensor(lengths)
            mfccs = tf.convert_to_tensor(mfccs)
            mels = tf.convert_to_tensor(mels)
            mags = tf.convert_to_tensor(mags)

            # Create Queues
            length, mfcc, mel, mag = tf.train.slice_input_producer([lengths, mfccs, mels, mags], shuffle=True)

            # Decoding
            mfcc = tf.py_func(lambda x: np.load(x), [mfcc], tf.float32)  # (T, n_mfccs)
            mel = tf.py_func(lambda x: np.load(x), [mel], tf.float32)  # (T,)
            mag = tf.py_func(lambda x: np.load(x), [mag], tf.float32)  # (T,)

            # exp
            def dicretize(mag, num_bins=100):
                bins = np.logspace(-1, 0, num_bins-1, base=10)
                mag = np.digitize(mag, bins).astype(np.int32)
                return mag

            # mag = tf.py_func(dicretize, [mag], tf.int32)

            # Set shapes
            mfcc.set_shape([None, hp.n_mfccs])
            mel.set_shape([None, hp.n_mels])
            mag.set_shape([None, hp.n_fft//2+1])

            # Create batch queues
            _, (mfccs, mels, mags) = tf.contrib.training.bucket_by_sequence_length(
                                                input_length=length,
                                                tensors=[mfcc, mel, mag],
                                                batch_size=hp.batch_size,
                                                bucket_boundaries=[i for i in range(minlen+1, maxlen-1, 50)],
                                                num_threads=16,
                                                capacity=hp.batch_size * 4,
                                                dynamic_pad=True)

            return mfccs, mels, mags, num_batch