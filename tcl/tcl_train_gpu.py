"""Training"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import numpy as np
from six.moves import xrange
import tensorflow as tf
#import tensorflow.compat.v1 as tf

import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tcl
from subfunc.showdata import *

FLAGS = tf.app.flags.FLAGS

# =============================================================
# =============================================================
def train(data,
          label,
          num_class,
          list_hidden_nodes,
          initial_learning_rate,
          momentum,
          max_steps,
          decay_steps,
          decay_factor,
          batch_size,
          train_dir,
          moving_average_decay = 0.9999,
          summary_steps = 500,
          checkpoint_steps = 10000,
          MLP_trainable = True,
          save_file = 'model.ckpt',
          load_file = None,
          random_seed = None):
    """Build and train a model
    Args:
        data: data. 2D ndarray [num_comp, num_data]
        label: labels. 1D ndarray [num_data]
        num_class: number of classes
        list_hidden_nodes: number of nodes for each layer. 1D array [num_layer]
        initial_learning_rate: initial learning rate
        momentum: momentum parameter (tf.train.MomentumOptimizer)
        max_steps: number of iterations (mini-batches)
        decay_steps: decay steps (tf.train.exponential_decay)
        decay_factor: decay factor (tf.train.exponential_decay)
        batch_size: mini-batch size
        train_dir: save directory
        moving_average_decay: (option) moving average decay of variables to be saved (tf.train.ExponentialMovingAverage)
        summary_steps: (option) interval to save summary
        checkpoint_steps: (option) interval to save checkpoint
        MLP_trainable: (option) If false, fix MLP layers
        save_file: (option) name of model file to save
        load_file: (option) name of model file to load
        random_seed: (option) random seed
    Returns:

    """

    with tf.Graph().as_default(): #, tf.device('/gpu:0'):

        # Set random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
            tf.set_random_seed(random_seed)

        global_step = tf.Variable(0, trainable=False)

        # create our batches
        data_holder, label_holder =  tf.train.shuffle_batch([tf.constant(data.T), tf.constant(label)], batch_size=batch_size, capacity=20*batch_size, min_after_dequeue=10*batch_size, enqueue_many = True)
        data_holder = tf.cast( data_holder, tf.float32 )
        label_holder = tf.cast( label_holder, tf.float32 )

        # Data holder
        #data_holder = tf.placeholder(tf.float32, shape=[None, data.shape[0]], name='data')
        #label_holder = tf.placeholder(tf.int32, shape=[None], name='label')

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits, feats = tcl.inference(data_holder, list_hidden_nodes, num_class, MLP_trainable=MLP_trainable)

        # Calculate loss.
        loss, accuracy = tcl.loss(logits, label_holder)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op, lr = tcl.train(loss,
                                 accuracy,
                                 global_step=global_step,
                                 initial_learning_rate=initial_learning_rate,
                                 momentum=momentum,
                                 decay_steps=decay_steps,
                                 decay_factor=decay_factor,
                                 moving_average_decay=moving_average_decay)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph.
        config = tf.ConfigProto( log_device_placement=False )
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(init)

        # Restore trained parameters from "load_file"
        if load_file is not None:
            #print("Load trainable parameters from {0:s}...".format(load_file))
            reader = tf.train.NewCheckpointReader(load_file)
            reader_var_to_shape_map = reader.get_variable_to_shape_map()
            #
            load_vars = tf.get_collection(FLAGS.FILTER_COLLECTION)
            # list up vars contained in the file
            initialized_vars = []
            for lv in load_vars:
                if reader_var_to_shape_map.has_key(lv.name.split(':')[0]):
                    #print("    {0:s}".format(lv.name))
                    initialized_vars.append(lv)
            # Restore
            saver_init = tf.train.Saver(initialized_vars)
            saver_init.restore(sess, load_file)


        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

        num_data = data.shape[1]
        num_steps_in_epoch = int(np.floor(num_data / batch_size))


        for step in xrange(max_steps):
            #start_time = time.time()

            #x_batch = tf.cast( x_batch, tf.float32)
            #y_batch = tf.cast( y_batch, tf.float32)

            # Make shuffled batch -----------------------------
            #if step % num_steps_in_epoch == 0:
            #    step_in_epoch = 0
            #    shuffle_idx = np.random.permutation(num_data)
            #x_batch = data[:, shuffle_idx[batch_size*step_in_epoch:batch_size*(step_in_epoch+1)]].T
            #y_batch = label[shuffle_idx[batch_size*step_in_epoch:batch_size*(step_in_epoch+1)]]
            #step_in_epoch = step_in_epoch + 1

            # Run ---------------------------------------------
            #feed_dict = {data_holder:x_batch, label_holder:y_batch}
            _, loss_value, accuracy_value, lr_value = sess.run([train_op, loss, accuracy, lr]) #, feed_dict=feed_dict)
            #duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % summary_steps == 0:
                summary_str = sess.run(summary_op)#, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % checkpoint_steps == 0:
                checkpoint_path = os.path.join(train_dir, save_file)
                saver.save(sess, checkpoint_path, global_step=step)


        # Save trained model ----------------------------------
        save_path = os.path.join(train_dir, save_file)
        #print("Save model in file: {0:s}".format(save_path))
        saver.save(sess, save_path)


