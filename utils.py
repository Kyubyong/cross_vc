# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/cross_vc
'''

from __future__ import print_function

from hparams import Hyperparams as hp
import librosa
import numpy as np
from scipy import signal
import copy
import tensorflow as tf

def get_mfcc_and_mag(fpath, trim=True):
    '''Returns normalized MFCCs and linear magnitude from `fpath`.
    Args:
      fpath: A string. The full path of a sound file.

    Returns:
      mfcc: A 2d array of shape (T, n_mfccs)
      mag: A 2d array of shape (T, 1+n_fft/2)
    '''
    # Loading sound file
    y, sr = librosa.load(fpath, sr=hp.sr)

    # Trimming
    if trim: y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=hp.n_fft,
                          hop_length=int(hp.sr * hp.frame_shift),
                          win_length=int(hp.sr * hp.frame_length))

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    # Get MFCCs
    mfcc = np.dot(librosa.filters.dct(hp.n_mfccs, mel.shape[0]), mel) # (n_mfcc, t)

    # Transpose for convenience
    mfcc = mfcc.T.astype(np.float32)  # (t, n_mfcc)
    mag = mag.T.astype(np.float32)
    return mfcc, mag

def spectrogram2wav(mag):
    '''# Generate wave file from magnitude spectrogram'''
    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag**hp.sharpening_factor)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)


def griffin_lim(spectrogram):
    '''Applies Griffin-Lim's raw.
    '''
    hop_length = int(hp.sr*hp.frame_shift)
    win_length = int(hp.sr*hp.frame_length)
    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hop_length, win_length=win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y


def invert_spectrogram(spectrogram):
    '''
    spectrogram: [f, t]
    '''
    hop_length = int(hp.sr * hp.frame_shift)
    win_length = int(hp.sr * hp.frame_length)
    return librosa.istft(spectrogram, hop_length, win_length=win_length, window="hann")

def learning_rate_decay(init_lr, global_step, warmup_steps=4000.):
    '''Noam scheme from tensor2tensor'''
    step = tf.to_float(global_step + 1)
    return init_lr * warmup_steps**0.5 * tf.minimum(step * warmup_steps**-1.5, step**-0.5)