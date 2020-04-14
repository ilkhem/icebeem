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
                computeApproxJacobian=False, apply_fastICA=True):
    ## define some variables:
    # random_seed = 0 # random seed
    # num_comp = 2 # number of components (dimension)
    # num_segment = 16 # number of segnents
    # num_segmentdata = 512 # number of data-points in each segment
    # num_layer = 2 # number of layers of mixing-MLP

    # MLP ---------------------------------------------------------
    # list_hidden_nodes = [num_comp *2]*(num_layer-1) + [num_comp]
    # list of the number of nodes of each hidden layer of feature-MLP
    # [layer1, layer2, ..., layer(num_layer)]

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
    # max_steps_init = int(7e4) # number of iterations (mini-batches) for initializing only MLR
    decay_steps_init = int(5e4)  # decay steps for initializing only MLR

    # Other -------------------------------------------------------
    # # Note: save folder must be under ./storage
    train_dir = '../storage/temp5'  # save directory
    saveparmpath = os.path.join(train_dir, 'parm.pkl')  # file name to save parameters

    num_segment = len(np.unique(label))

    ## generate/load some data
    # sensor, source, label = generate_artificial_data(num_comp=num_comp, num_segment=num_segment, num_segmentdata=num_segmentdata, num_layer=num_layer, random_seed=random_seed)

    # Preprocessing -----------------------------------------------
    sensor, pca_parm = pca(sensor, num_comp=num_comp)

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
    apply_fast_ica = True
    eval_dir = '../storage/temp5'
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
    if apply_fast_ica:
        ica = FastICA(random_state=random_seed)
        feat_val_ica = ica.fit_transform(feat_val)
        feateval_ica = feat_val_ica.T  # Estimated feature
    else:
        feateval_ica = feat_val.T

    featval = feat_val.T

    return featval, feateval_ica, accuracy

#
#
#
# ## now that we have trained everything, we can evaluate results:
#
# #apply_fastICA = True
# nonlinearity_to_source = 'abs'
# eval_dir = '../storage/temp5'
# parmpath = os.path.join(eval_dir, 'parm.pkl')
# ckpt = tf.train.get_checkpoint_state(eval_dir)
# modelpath = ckpt.model_checkpoint_path
#
#
# with tf.Graph().as_default() as g:
#
#     data_holder = tf.placeholder(tf.float32, shape=[None, sensor.shape[0]], name='data')
#     label_holder = tf.placeholder(tf.int32, shape=[None], name='label')
#
#     # Build a Graph that computes the logits predictions from the
#     # inference model.
#     logits, feats = tcl.inference(data_holder, list_hidden_nodes, num_class=num_segment)
#
#     # Calculate predictions.
#     top_value, preds = tf.nn.top_k(logits, k=1, name='preds')
#
#     # Restore the moving averaged version of the learned variables for eval.
#     variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay)
#     variables_to_restore = variable_averages.variables_to_restore()
#     saver = tf.train.Saver(variables_to_restore)
#
#     with tf.Session() as sess:
#         saver.restore(sess, ckpt.model_checkpoint_path)
#
#         tensor_val = tf_eval.get_tensor(sensor, [preds, feats], sess, data_holder, batch=256)
#         pred_val = tensor_val[0].reshape(-1)
#         feat_val = tensor_val[1]
#
#
# # Calculate accuracy ------------------------------------------
# accuracy, confmat = tf_eval.calc_accuracy(pred_val, label)
#
# # Apply fastICA -----------------------------------------------
# if apply_fastICA:
#     ica = FastICA(random_state=random_seed)
#     feat_val_ica = ica.fit_transform(feat_val)
# else:
# 	feat_val_ica = feat_val
#
#
# # Evaluate ----------------------------------------------------
# #if nonlinearity_to_source == 'abs':
# #    xseval = np.abs(source) # Original source
# #else:
# #    raise ValueError
# feateval_ica = feat_val_ica.T # Estimated feature
# featval = feat_val.T
# #
# #corrmat, sort_idx, _ = tf_eval.correlation(feateval, xseval, 'Pearson')
# #meanabscorr = np.mean(np.abs(np.diag(corrmat)))
#
# return featval, feateval_ica, accuracy
#
