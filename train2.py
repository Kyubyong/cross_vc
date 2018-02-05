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
from utils import *

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

            merged = sess.run(g.merged)
            writer.add_summary(merged, global_step=gs)

            # Save
            saver2.save(sess, logdir + '/model_gs'.format(gs))

        writer.close()
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    train2(); print("Done")
