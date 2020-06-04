### Wrapper function for TCL 
#
# this code is adapted from: https://github.com/hirosm/TCL
#
#
import os

import numpy as np
import tensorflow as tf
from sklearn.decomposition import FastICA

from .tcl_core import inference
from .tcl_core import train_gpu as train
from .tcl_eval import get_tensor, calc_accuracy
from .tcl_preprocessing import pca

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def TCL_wrapper(sensor, label, list_hidden_nodes, random_seed=0, max_steps=int(7e4), max_steps_init=int(7e4),
                ckpt_dir='./', test=False):
    # Training ----------------------------------------------------
    initial_learning_rate = 0.01  # initial learning rate
    momentum = 0.9  # momentum parameter of SGD
    # max_steps = int(7e4) # number of iterations (mini-batches)
    decay_steps = int(5e4)  # decay steps (tf.train.exponential_decay)
    decay_factor = 0.1  # decay factor (tf.train.exponential_decay)
    batch_size = 512  # mini-batch size
    moving_average_decay = 0.9999  # moving average decay of variables to be saved
    checkpoint_steps = 1e5  # interval to save checkpoint
    num_comp = sensor.shape[0]

    # for MLR initialization
    decay_steps_init = int(5e4)  # decay steps for initializing only MLR

    # Other -------------------------------------------------------
    train_dir = ckpt_dir  # save directory

    num_segment = len(np.unique(label))

    # Preprocessing -----------------------------------------------
    sensor, pca_parm = pca(sensor, num_comp=num_comp)

    if not test:
        # Train model (only MLR) --------------------------------------
        train(sensor,
              label,
              num_class=len(np.unique(label)),  # num_segment,
              list_hidden_nodes=list_hidden_nodes,
              initial_learning_rate=initial_learning_rate,
              momentum=momentum,
              max_steps=max_steps_init,  # For init
              decay_steps=decay_steps_init,  # For init
              decay_factor=decay_factor,
              batch_size=batch_size,
              train_dir=train_dir,
              checkpoint_steps=checkpoint_steps,
              moving_average_decay=moving_average_decay,
              MLP_trainable=False,  # For init
              save_file='model_init.ckpt',  # For init
              random_seed=random_seed)

        init_model_path = os.path.join(train_dir, 'model_init.ckpt')

        # Train model -------------------------------------------------
        train(sensor,
              label,
              num_class=len(np.unique(label)),  # num_segment,
              list_hidden_nodes=list_hidden_nodes,
              initial_learning_rate=initial_learning_rate,
              momentum=momentum,
              max_steps=max_steps,
              decay_steps=decay_steps,
              decay_factor=decay_factor,
              batch_size=batch_size,
              train_dir=train_dir,
              checkpoint_steps=checkpoint_steps,
              moving_average_decay=moving_average_decay,
              load_file=init_model_path,
              random_seed=random_seed)

    # now that we have trained everything, we can evaluate results:
    eval_dir = ckpt_dir
    ckpt = tf.train.get_checkpoint_state(eval_dir)

    with tf.Graph().as_default():
        data_holder = tf.placeholder(tf.float32, shape=[None, sensor.shape[0]], name='data')

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits, feats = inference(data_holder, list_hidden_nodes, num_class=num_segment)

        # Calculate predictions.
        top_value, preds = tf.nn.top_k(logits, k=1, name='preds')

        # Restore the moving averaged version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            saver.restore(sess, ckpt.model_checkpoint_path)

            tensor_val = get_tensor(sensor, [preds, feats], sess, data_holder, batch=256)
            pred_val = tensor_val[0].reshape(-1)
            feat_val = tensor_val[1]

    # Calculate accuracy ------------------------------------------
    accuracy, confmat = calc_accuracy(pred_val, label)

    # Apply fastICA -----------------------------------------------
    ica = FastICA(random_state=random_seed)
    feat_val_ica = ica.fit_transform(feat_val)

    feat_val_ica = feat_val_ica.T  # Estimated feature
    feat_val = feat_val.T

    return feat_val, feat_val_ica, accuracy
