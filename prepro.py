# -*- coding: utf-8 -*-
# #/usr/bin/python2

'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/deepvoice3
'''

import numpy as np

from hparams import Hyperparams as hp
import glob
import os
import tqdm
from utils import load_vocab, get_audio_features, timit2arpa
import codecs

if __name__ == "__main__":
    phn2idx, idx2phn = load_vocab()

    # print("Working on TIMIT")
    # for f in tqdm.tqdm(glob.glob('/data/public/rw/timit/TIMIT/*/*/*/*.wav')):
    #     fname = "_".join(f.split("/")[-2:]).replace(".wav", ".npy")
    #     try:
    #         mfcc, _, _ = get_audio_features(f, sr=16000, trim=False)  # (T, n_mfccs) (T, n_mels), (T, 1+n_fft/2) float32
    #     except:
    #         continue
    #
    #     f = f.replace("WAV.wav", "PHN").replace("wav", 'PHN')
    #     phns = np.zeros(shape=(mfcc.shape[0],), dtype=np.int32)
    #     for line in open(f, 'r'):
    #         start_point, _, phn = line.strip().split()
    #         if 'cl' in phn: continue
    #         phn = timit2arpa.get(phn, phn)
    #         bnd = int(start_point) // hp.hop_length
    #         # Note that the unit of start point and hop_length is sample.
    #         # So bnd is the ordering number of frames
    #         phns[bnd:] = phn2idx[phn]
    #
    #     if not os.path.exists("timit/mfccs"): os.makedirs("timit/mfccs")
    #     if not os.path.exists("timit/phns"): os.makedirs("timit/phns")
    #     np.save("timit/mfccs/" + fname, mfcc)
    #     np.save("timit/phns/" + fname, phns)


    print("Working on nick")
    for f in tqdm.tqdm(glob.glob("/data/private/voice/nick/*/*.wav")):
    # for f in ("/data/private/voice/nick/Tom/Tom_01-033.wav",):
        fname = os.path.basename(f)
        try:
            mfcc, mel, mag = get_audio_features(f, sr=hp.sr, trim=False)  # (T, n_mfccs) (T, n_mels), (T, 1+n_fft/2) float32
            if not os.path.exists("nick/mfccs"): os.makedirs("nick/mfccs")
            if not os.path.exists("nick/mels"): os.makedirs("nick/mels")
            if not os.path.exists("nick/mags"): os.makedirs("nick/mags")
            if len(mfcc) > hp.max_len: continue

            np.save("nick/mfccs/" + fname, mfcc)
            np.save("nick/mels/" + fname, mel)
            np.save("nick/mags/" + fname, mag)
        except:
            continue

        try:
            phn_name = f.replace("WAV.wav", "PHN").replace("wav", 'PHN')
            if os.path.exists(phn_name):
                phns = np.zeros(shape=(mfcc.shape[0],), dtype=np.int32)
                for line in open(phn_name, 'r'):
                    start_point, _, phn = line.strip().split() # !unit: ms
                    # bnd = int(start_point) // int(hp.frame_shift*1000)
                    bnd = int(start_point) * hp.sr / hp.hop_length // 1000
                    phn = phn.replace("h#", "sil")
                    phns[bnd:] = phn2idx[phn]

                if not os.path.exists("nick/phns"): os.makedirs("nick/phns")
                np.save("nick/phns/" + fname, phns)
        except:
            continue

    # print("Working on {}".format(hp.data))
    #
    # for line in codecs.open('/data/private/voice/{}/transcript.csv'.format(hp.data), 'r', 'utf8'):
    #     wav, text1, text2, label = line.strip().split("|")
    #     if label == "0":
    #         wav = '/data/private/voice/{}/'.format(hp.data) + wav
    #         fname = os.path.basename(wav).replace(".wav", ".npy")
    #
    #         mfcc, mel, mag = get_audio_features(wav, sr=hp.sr, trim=False) # (T, n_mfccs) (T, n_mels), (T, 1+n_fft/2) float32
    #         if not os.path.exists("{}/mfccs".format(hp.data)): os.makedirs("{}/mfccs".format(hp.data))
    #         if not os.path.exists("{}/mels".format(hp.data)): os.makedirs("{}/mels".format(hp.data))
    #         if not os.path.exists("{}/mags".format(hp.data)): os.makedirs("{}/mags".format(hp.data))
    #         # if len(mfcc) > 800: continue
    #         np.save("{}/mfccs/".format(hp.data) + fname, mfcc)
    #         np.save("{}/mels/".format(hp.data) + fname, mel)
    #         np.save("{}/mags/".format(hp.data) + fname, mag)


