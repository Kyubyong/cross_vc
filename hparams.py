# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/cross_vc
'''

class Hyperparams:
    # data
    timit = "/data/public/rw/timit/TIMIT/*/*/*/*.wav"
    arctic = "/data/public/rw/arctic/slt/*.wav"
    test_data = "50lang/*.wav"
    vocab = ['h#', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl',
            'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi',
            'er', 'ey', 'f', 'g', 'gcl', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh',
            'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl',
            'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh',
             ('s', 'tcl', 't'), ('ix', 'kcl', 'k'), ('kcl', 'k', 's'), ('gcl', 'g', 'r'), ('l', 'ay', 'kcl'),
             ('n', 'tcl', 't'), ('dcl', 'd', 'aa'), ('g', 'r', 'iy'), ('aa', 'r', 'kcl'), ('tcl', 't', 'ix'),
             ('r', 'ae', 'gcl'), ('s', 'kcl', 'k'), ('d', 'aa', 'r'), ('dh', 'ae', 'tcl'), ('r', 'kcl', 'k'),
             ('kcl', 'k', 'eh'), ('ao', 'l', 'y'), ('r', 'iy', 's'), ('sh', 'epi', 'w'), ('gcl', 'g', 'l'),
             ('l', 'y', 'ih'), ('ae', 'gcl', 'g'), ('sh', 'iy', 'hv'), ('axr', 'dcl', 'd'), ('tcl', 't', 'r'),
             ('w', 'ao', 'dx'), ('k', 'eh', 'r'), ('s', 'pcl', 'p'), ('k', 's', 'ux'), ('l', 'iy', 'r'),
             ('g', 'l', 'ay'), ('iy', 'r', 'ae'), ('q', 'ao', 'l'), ('ae', 's', 'kcl'), ('pcl', 'p', 'r'),
             ('n', 'dcl', 'd'), ('epi', 'w', 'ao'), ('eh', 'r', 'iy'), ('y', 'ih', 'axr'), ('dx', 'ix', 'kcl'),
             ('iy', 's', 'iy'), ('hv', 'ae', 'dcl'), ('kcl', 'k', 'ix'), ('oy', 'l', 'iy'), ('s', 'iy', 'w'),
             ('ay', 'kcl', 'dh'), ('s', 'ux', 'tcl'), ('dcl', 'd', 'ix'), ('r', 'iy', 'ix'), ('kcl', 'dh', 'ae'),
             ('iy', 'ix', 'n'), ('iy', 'hv', 'ae'), ('ix', 'n', 'tcl'), ('pcl', 'p', 'l'), ('ao', 'dx', 'axr'),
             ('iy', 'tcl', 't'), ('d', 'ow', 'n'), ('tcl', 't', 'ux'), ('ay', 'kcl', 'k'), ('n', 'gcl', 'g'),
             ('s', 'epi', 'm'), ('ix', 'tcl', 't'), ('w', 'aa', 'sh'), ('m', 'iy', 'dx'), ('kcl', 'k', 'l'),
             ('s', 'kcl', 'm'), ('ax', 'bcl', 'b'), ('iy', 'kcl', 'k'), ('m', 'pcl', 'p'), ('iy', 'dx', 'ix'),
             ('dcl', 'd', 'ih'), ('aa', 'sh', 'epi'), ('ih', 'kcl', 'k'), ('nx', 'ae', 's'), ('q', 'ix', 'n'),
             ('d', 'ow', 'nx'), ('ow', 'nx', 'ae'), ('ng', 'gcl', 'g'), ('ix', 'n', 'q'), ('kcl', 'k', 'r'),
             ('ix', 'dcl', 'd'), ('bcl', 'b', 'iy'), ('q', 'oy', 'l'), ('tcl', 't', 's'), ('dx', 'axr', 'q'),
             ('pcl', 'p', 'axr'), ('kcl', 'k', 'w'), ('w', 'ao', 'sh'), ('axr', 'q', 'ao'), ('iy', 'w', 'aa'),
             ('ix', 'gcl', 'g'), ('ax', 'pcl', 'p'), ('iy', 'w', 'ao'), ('iy', 'pcl', 'p'), ('tcl', 't', 'iy'),
             ('er', 'dcl', 'd'), ('kcl', 'k', 'ae'), ('kcl', 'm', 'iy'), ('sh', 'ix', 'n'), ('kcl', 'k', 'el')]


    # signal processing
    sr = 16000  # Sampling rate.
    n_fft = 512  # fft points (samples)
    frame_length = 0.025
    frame_shift = frame_length / 5

    n_mels = 80  # Number of Mel banks to generate
    n_mfccs = 40 # Number of MFCCs
    sharpening_factor = 1.4  # Exponent for amplifying the predicted magnitude
    n_iter = 50  # Number of inversion iterations
    preemphasis = .97  # or None
    max_db = 100
    ref_db = 20
    maxlen = 400 # frames

    # model
    hidden_units = 256  # alias = E
    num_banks = 16
    num_highwaynet_blocks = 4
    dropout_rate = 0.05

    # training
    batch_size = 32
    lr = 0.0005 # learning rate.
    logdir = "logdir/triphone" #lr=0.002
