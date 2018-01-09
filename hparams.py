# -*- coding: utf-8 -*-
#/usr/bin/python2

class Hyperparams:
    # signal processing # Timit
    # sr = 16000  # Sampling rate.
    # n_fft = 2048  # fft points (samples)
    # frame_shift = 0.005  # seconds
    # frame_length = 0.025  # seconds
    # hop_length = int(sr * frame_shift)  # samples  This is dependent on the frame_shift.
    # win_length = int(sr * frame_length)  # samples This is dependent on the frame_length.

    data = "nick"
    sr = 22050  # Sampling rate.
    n_fft = 2048  # fft points (samples)
    hop_length = 512 // 4  # samples  This is dependent on the frame_shift.
    win_length = 512  # samples This is dependent on the frame_length.
    frame_shift = float(hop_length) / sr   # seconds
    frame_length = float(win_length) / sr  # seconds

    n_mels = 80  # Number of Mel banks to generate
    n_mfccs = 40
    sharpening_factor = 1.4  # Exponent for amplifying the predicted magnitude
    n_iter = 50  # Number of inversion iterations
    preemphasis = .97  # or None
    max_db = 100
    ref_db = 20
    max_len = 800 # frames

    # model
    hidden_units = 256  # alias = E
    num_banks = 16
    num_highwaynet_blocks = 4
    norm_type = 'ln'  # a normalizer function. value: bn, ln, ins, or None
    dropout_rate = 0.05

    # training
    batch_size = 16
    lr = 0.001
    logdir = "logdir/50"
# from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
# print_tensors_in_checkpoint_file(file_name='logdir/040/train1/model_gs_13158', tensor_name='', all_tensors=False)