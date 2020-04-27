"""tcl"""

import os
import os.path

import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

FILTER_COLLECTION = 'filter'


# =============================================================
# =============================================================
def _variable_init(name, shape, wd, initializer=tf.contrib.layers.variance_scaling_initializer(), trainable=True,
                   collections=None):
    """Helper to create an initialized Variable with weight decay.

    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
    Returns:
        Variable Tensor
    """

    if collections is None:
        collections = [tf.GraphKeys.GLOBAL_VARIABLES]
    else:
        collections = [tf.GraphKeys.GLOBAL_VARIABLES] + collections

    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32, trainable=trainable,
                              collections=collections)

    # Weight decay
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    return var


# =============================================================
# =============================================================
def inference(x, list_hidden_nodes, num_class, wd=1e-4, maxout_k=2, MLP_trainable=True, feature_nonlinearity='abs'):
    """Build the model.
        MLP4 with maxout activation units
    Args:
        x: data holder.
        list_hidden_nodes: number of nodes for each layer. 1D array [num_layer]
        num_class: number of classes of MLR
        wd: (option) parameter of weight decay (not for bias)
        maxout_k: (option) number of affine feature maps
        MLP_trainable: (option) If false, fix MLP4 layers
        feature_nonlinearity: (option) Nonlinearity of the last hidden layer (feature value)
    Returns:
        logits: logits tensor:
        feat: feature tensor
    """

    print("Building model...")

    # Maxout --------------------------------------------------
    def maxout(y, k):
        #   y: data tensor
        #   k: number of affine feature maps
        input_shape = y.get_shape().as_list()
        ndim = len(input_shape)
        ch = input_shape[-1]
        assert ndim == 4 or ndim == 2
        assert ch is not None and ch % k == 0
        if ndim == 4:
            y = tf.reshape(y, [-1, input_shape[1], input_shape[2], ch / k, k])
        else:
            y = tf.reshape(y, [-1, int(ch / k), k])
        y = tf.reduce_max(y, ndim)
        return y

    num_layer = len(list_hidden_nodes)

    # Hidden layers -------------------------------------------
    for ln in range(num_layer):
        with tf.variable_scope('layer' + str(ln + 1)) as scope:
            in_dim = list_hidden_nodes[ln - 1] if ln > 0 else x.get_shape().as_list()[1]
            out_dim = list_hidden_nodes[ln]

            if ln < num_layer - 1:  # Increase number of nodes for maxout
                out_dim = maxout_k * out_dim

            # Inner product
            W = _variable_init('W', [in_dim, out_dim], wd, trainable=MLP_trainable,
                               collections=[FILTER_COLLECTION])
            b = _variable_init('b', [out_dim], 0, tf.constant_initializer(0.0), trainable=MLP_trainable,
                               collections=[FILTER_COLLECTION])
            x = tf.nn.xw_plus_b(x, W, b)

            # Nonlinearity
            if ln < num_layer - 1:
                x = maxout(x, maxout_k)
            else:  # The last layer (feature value)
                if feature_nonlinearity == 'abs':
                    x = tf.abs(x)
                else:
                    raise ValueError

            # Add summary
            tf.summary.histogram('layer' + str(ln + 1) + '/activations', x)

    feats = x

    # MLR -----------------------------------------------------
    with tf.variable_scope('MLR') as scope:
        in_dim = list_hidden_nodes[-1]
        out_dim = num_class

        # Inner product
        W = _variable_init('W', [in_dim, out_dim], wd, collections=[FILTER_COLLECTION])
        b = _variable_init('b', [out_dim], 0, tf.constant_initializer(0.0), collections=[FILTER_COLLECTION])
        logits = tf.nn.xw_plus_b(x, W, b)

    return logits, feats


# =============================================================
# =============================================================
def tcl_loss(logits, labels):
    """Add L2Loss to all the trainable variables.
        Add summary for "Loss" and "Loss/avg".
    Args:
        logits: logits from inference().
        labels: labels from distorted_inputs or inputs(). 1-D tensor
                of shape [batch_size]
    Returns:
        Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # Calculate accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='acurracy')

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss'), accuracy


# =============================================================
# =============================================================
def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.
        Generates moving average for all losses and associated summaries for
        visualizing the performance of the network.

    Args:
        total_loss: total loss from loss().
    Returns:
        loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


# =============================================================
# =============================================================
def train(total_loss,
          accuracy,
          global_step,
          initial_learning_rate,
          momentum,
          decay_steps,
          decay_factor,
          moving_average_decay=0.9999,
          moving_average_collections=tf.trainable_variables()):
    """Train model.
        Create an optimizer and apply to all trainable variables. Add moving
        average for all trainable variables.
    Args:
        total_loss: total loss from loss().
        accuracy: accuracy tensor
        global_step: integer variable counting the number of training steps processed.
        initial_learning_rate: initial learning rate
        momentum: momentum parameter (tf.train.MomentumOptimizer)
        decay_steps: decay steps (tf.train.exponential_decay)
        decay_factor: decay factor (tf.train.exponential_decay)
        moving_average_decay: (option) moving average decay of variables to be saved
    Returns:
        train_op: op for training.
    """

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    decay_factor,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Generate moving averages of accuracy and associated summaries.
    accu_averages = tf.train.ExponentialMovingAverage(0.9, name='avg_accu')
    accu_averages_op = accu_averages.apply([accuracy])
    tf.summary.scalar(accuracy.op.name + ' (raw)', accuracy)
    tf.summary.scalar(accuracy.op.name, accu_averages.average(accuracy))

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op, accu_averages_op]):
        # opt = tf.train.GradientDescentOptimizer(lr)
        opt = tf.train.MomentumOptimizer(lr, momentum)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op, lr


# =============================================================
# =============================================================
def train_cpu(data,
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
              moving_average_decay=0.9999,
              summary_steps=500,
              checkpoint_steps=10000,
              MLP_trainable=True,
              save_file='model.ckpt',
              load_file=None,
              random_seed=None):
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
        MLP_trainable: (option) If false, fix MLP4 layers
        save_file: (option) name of model file to save
        load_file: (option) name of model file to load
        random_seed: (option) random seed
    Returns:

    """

    with tf.Graph().as_default(), tf.device('/cpu:0'):

        # Set random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
            tf.set_random_seed(random_seed)

        global_step = tf.Variable(0, trainable=False)

        # create our batches
        data_holder, label_holder = tf.train.shuffle_batch([tf.constant(data.T), tf.constant(label)],
                                                           batch_size=batch_size, capacity=20 * batch_size,
                                                           min_after_dequeue=10 * batch_size, enqueue_many=True)
        data_holder = tf.cast(data_holder, tf.float32)
        label_holder = tf.cast(label_holder, tf.float32)

        # Data holder
        # data_holder = tf.placeholder(tf.float32, shape=[None, data.shape[0]], name='data')
        # label_holder = tf.placeholder(tf.int32, shape=[None], name='label')

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits, feats = inference(data_holder, list_hidden_nodes, num_class, MLP_trainable=MLP_trainable)

        # Calculate loss.
        loss, accuracy = tcl_loss(logits, label_holder)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op, lr = train(loss,
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
        config = tf.ConfigProto(log_device_placement=False)
        # config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(init)

        # Restore trained parameters from "load_file"
        if load_file is not None:
            print("Load trainable parameters from {0:s}...".format(load_file))
            reader = tf.train.NewCheckpointReader(load_file)
            reader_var_to_shape_map = reader.get_variable_to_shape_map()
            #
            load_vars = tf.get_collection(FILTER_COLLECTION)
            # list up vars contained in the file
            initialized_vars = []
            for lv in load_vars:
                if lv.name.split(':')[0] in reader_var_to_shape_map:
                    print("    {0:s}".format(lv.name))
                    initialized_vars.append(lv)
            # Restore
            saver_init = tf.train.Saver(initialized_vars)
            saver_init.restore(sess, load_file)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

        num_data = data.shape[1]
        num_steps_in_epoch = int(np.floor(num_data / batch_size))

        for step in range(max_steps):
            # start_time = time.time()

            # x_batch = tf.cast( x_batch, tf.float32)
            # y_batch = tf.cast( y_batch, tf.float32)

            # Make shuffled batch -----------------------------
            # if step % num_steps_in_epoch == 0:
            #    step_in_epoch = 0
            #    shuffle_idx = np.random.permutation(num_data)
            # x_batch = data[:, shuffle_idx[batch_size*step_in_epoch:batch_size*(step_in_epoch+1)]].T
            # y_batch = label[shuffle_idx[batch_size*step_in_epoch:batch_size*(step_in_epoch+1)]]
            # step_in_epoch = step_in_epoch + 1

            # Run ---------------------------------------------
            # feed_dict = {data_holder:x_batch, label_holder:y_batch}
            _, loss_value, accuracy_value, lr_value = sess.run([train_op, loss, accuracy, lr])  # , feed_dict=feed_dict)
            # duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % summary_steps == 0:
                # print(step)
                summary_str = sess.run(summary_op)  # , feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % checkpoint_steps == 0:
                checkpoint_path = os.path.join(train_dir, save_file)
                saver.save(sess, checkpoint_path, global_step=step)

        # Save trained model ----------------------------------
        save_path = os.path.join(train_dir, save_file)
        print("Save model in file: {0:s}".format(save_path))
        saver.save(sess, save_path)


def train_gpu(data,
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
              moving_average_decay=0.9999,
              summary_steps=500,
              checkpoint_steps=10000,
              MLP_trainable=True,
              save_file='model.ckpt',
              load_file=None,
              random_seed=None):
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
        MLP_trainable: (option) If false, fix MLP4 layers
        save_file: (option) name of model file to save
        load_file: (option) name of model file to load
        random_seed: (option) random seed
    Returns:

    """

    with tf.Graph().as_default():  # , tf.device('/gpu:0'):

        # Set random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
            tf.set_random_seed(random_seed)

        global_step = tf.Variable(0, trainable=False)

        # create our batches
        data_holder, label_holder = tf.train.shuffle_batch([tf.constant(data.T), tf.constant(label)],
                                                           batch_size=batch_size, capacity=20 * batch_size,
                                                           min_after_dequeue=10 * batch_size, enqueue_many=True)
        data_holder = tf.cast(data_holder, tf.float32)
        label_holder = tf.cast(label_holder, tf.float32)

        # Data holder
        # data_holder = tf.placeholder(tf.float32, shape=[None, data.shape[0]], name='data')
        # label_holder = tf.placeholder(tf.int32, shape=[None], name='label')

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits, feats = inference(data_holder, list_hidden_nodes, num_class, MLP_trainable=MLP_trainable)

        # Calculate loss.
        loss, accuracy = tcl_loss(logits, label_holder)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op, lr = train(loss,
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
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(init)

        # Restore trained parameters from "load_file"
        if load_file is not None:
            print("Load trainable parameters from {0:s}...".format(load_file))
            reader = tf.train.NewCheckpointReader(load_file)
            reader_var_to_shape_map = reader.get_variable_to_shape_map()
            #
            load_vars = tf.get_collection(FILTER_COLLECTION)
            # list up vars contained in the file
            initialized_vars = []
            for lv in load_vars:
                if lv.name.split(':')[0] in reader_var_to_shape_map:
                    print("    {0:s}".format(lv.name))
                    initialized_vars.append(lv)
            # Restore
            saver_init = tf.train.Saver(initialized_vars)
            saver_init.restore(sess, load_file)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

        num_data = data.shape[1]
        num_steps_in_epoch = int(np.floor(num_data / batch_size))

        for step in range(max_steps):
            # start_time = time.time()

            # x_batch = tf.cast( x_batch, tf.float32)
            # y_batch = tf.cast( y_batch, tf.float32)

            # Make shuffled batch -----------------------------
            # if step % num_steps_in_epoch == 0:
            #    step_in_epoch = 0
            #    shuffle_idx = np.random.permutation(num_data)
            # x_batch = data[:, shuffle_idx[batch_size*step_in_epoch:batch_size*(step_in_epoch+1)]].T
            # y_batch = label[shuffle_idx[batch_size*step_in_epoch:batch_size*(step_in_epoch+1)]]
            # step_in_epoch = step_in_epoch + 1

            # Run ---------------------------------------------
            # feed_dict = {data_holder:x_batch, label_holder:y_batch}
            _, loss_value, accuracy_value, lr_value = sess.run([train_op, loss, accuracy, lr])  # , feed_dict=feed_dict)
            # duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % summary_steps == 0:
                # print(step)
                summary_str = sess.run(summary_op)  # , feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % checkpoint_steps == 0:
                checkpoint_path = os.path.join(train_dir, save_file)
                saver.save(sess, checkpoint_path, global_step=step)

        # Save trained model ----------------------------------
        save_path = os.path.join(train_dir, save_file)
        print("Save model in file: {0:s}".format(save_path))
        saver.save(sess, save_path)
