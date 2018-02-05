# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/cross_vc
'''

from __future__ import print_function
from hparams import Hyperparams as hp
from tqdm import tqdm
from graph import Graph
import tensorflow as tf
from data_load import load_data
import numpy as np

def train1():
    g = Graph(); print("Training Graph loaded")
    logdir = hp.logdir + "/train1"

    # Session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Restore saved variables
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net1') + \
                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'training')
        saver = tf.train.Saver(var_list=var_list)
        ckpt = tf.train.latest_checkpoint(logdir)
        if ckpt is not None: saver.restore(sess, ckpt)

        # Writer & Queue
        writer = tf.summary.FileWriter(logdir, sess.graph)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Inspect variables
        saver.save(sess, logdir + '/model_gs_0')
        from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
        print_tensors_in_checkpoint_file(file_name=hp.logdir+'/train1/model_gs_0', tensor_name='', all_tensors=False)

        # Training
        while 1:
            for _ in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                gs, _ = sess.run([g.global_step, g.train_op])

            # Write checkpoint files at every epoch
            merged = sess.run(g.merged)
            writer.add_summary(merged, global_step=gs)

            # evaluation
            with tf.Graph().as_default(): eval1()

            # Save
            saver.save(sess, logdir + '/model_gs_{}'.format(gs))

            if gs > 10000: break

        writer.close()
        coord.request_stop()
        coord.join(threads)

def eval1():
    # Load data
    mfccs, phns = load_data(mode="eval1")

    # Graph
    g = Graph("eval1"); print("Evaluation Graph loaded")
    logdir = hp.logdir + "/train1"

    # Session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Restore saved variables
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'net1') +\
                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'training')
        saver = tf.train.Saver(var_list=var_list)

        ckpt = tf.train.latest_checkpoint(logdir)
        if ckpt is not None: saver.restore(sess, ckpt)

        # Writer
        writer = tf.summary.FileWriter(logdir, sess.graph)

        # Evaluation
        merged, gs = sess.run([g.merged, g.global_step], {g.mfccs: mfccs, g.phones: phns})

        #  Write summaries
        writer.add_summary(merged, global_step=gs)
        writer.close()

if __name__ == '__main__':
    train1(); print("Done")
