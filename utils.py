# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import print_function

from hparams import Hyperparams as hp
import librosa
import numpy as np
from scipy import signal
import copy

timit2arpa = {"h#":"sil",
              "pau":"sil",
              "epi":"sil",
              "el":"l",
              "en":"n",
              "ix":"ih",
              "ax":"ah",
              "em":"m",
              "eng":"ng",
              "nx":"n",
              "axr":"er",
              "hv":"hh",
              "dx":"d",
              "ax-h":"ah",
              "ux":"uw",
              "q":"t"}

def load_vocab():
    # phns = ['PAD', 'h#', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl',
    #         'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi',
    #         'er', 'ey', 'f', 'g', 'gcl', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh',
    #         'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl',
    #         'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']

    phns = ['sil', 'aa', 'ae', 'ah', 'ao', 'aw', 'ay', 'b', 'ch', 'd', 'dh', 'eh', 'er', 'ey', 'f', 'g',
            'hh', 'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 't',
            'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh']

    phn2idx = {phn: idx for idx, phn in enumerate(phns)}
    idx2phn = {idx: phn for idx, phn in enumerate(phns)}

    return phn2idx, idx2phn

def get_audio_features(sound_file, sr=None, trim=False, preemphasis=True):
    '''Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
    Args:
      sound_file: A string. The full path of a sound file.

    Returns:
      mel: A 2d array of shape (T, n_mels) <- Transposed
      mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
    '''
    # Loading sound file
    y, sr = librosa.load(sound_file, sr=sr)

    # Trimming
    if trim: y, _ = librosa.effects.trim(y)

    # Preemphasis
    if preemphasis: y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # Get MFCCs
    mfcc = np.dot(librosa.filters.dct(hp.n_mfccs, mel.shape[0]), mel) # (n_mfcc, T)

    # normalize
    mel = np.clip((mel + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    # Transpose
    mfcc = mfcc.T.astype(np.float32)  # (T, n_mfcc)
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mfcc, mel, mag



def spectrogram2wav(mag):
    '''# Generate wave file from spectrogram'''
    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # sharpening factor
    mag **= hp.sharpening_factor

    # wav reconstruction
    wav = griffin_lim(mag)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav

def griffin_lim(spectrogram):
    '''Applies Griffin-Lim's raw.
    '''
    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y

def invert_spectrogram(spectrogram):
    '''
    spectrogram: [f, t]
    '''
    return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")

# def plot_alignment(alignments, gs):
#     """Plots the alignment
#     alignments: A list of (numpy) matrix of shape (encoder_steps, decoder_steps)
#     gs : (int) global step
#     """
#     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
#     im = ax.imshow(alignments[0])
#     ax.axis('off')
#
#     cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#     fig.colorbar(im, cax=cbar_ax)
#     plt.suptitle('{} Steps'.format(gs))
#     plt.savefig('{}/alignment_{}.png'.format(hp.logdir, gs), format='png')

def learning_rate_decay(init_lr, global_step):
    # Noam scheme from tensor2tensor:
    warmup_steps = 4000.0
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps**0.5 * tf.minimum(step * warmup_steps**-1.5, step**-0.5)

# # LJ/mfccs/wavs_LJ008-0150.npy
# mfcc, mel, mag = get_audio_features('/data/private/voice/LJSpeech-1.0/wavs/LJ008-0150.wav')
# print(mag.shape)
# wav=spectrogram2wav(mag)
# from scipy.io.wavfile import write
# write("t.wav", hp.sr, wav)
#
# mag = np.load('LJ/mags/wavs_LJ008-0150.npy')
# print(mag.shape)
# wav=spectrogram2wav(mag)
# from scipy.io.wavfile import write
# write("t2.wav", hp.sr, wav)