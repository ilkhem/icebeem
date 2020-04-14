""" Fuctions for evaluation
    This software includes the work that is distributed in the Apache License 2.0
"""

import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix


# =============================================================
# =============================================================
def get_tensor(x, vars, sess, data_holder, batch=256):
    """Get tensor data .
    Args:
        x: input data [Ndim, Ndata]
        vars: tensors (list)
        sess: session
        data_holder: data holder
        batch: batch size
    Returns:
        y: value of tensors
    """

    Ndata = x.shape[1]
    if batch is None:
        Nbatch = Ndata
    else:
        Nbatch = batch
    Niter = int(np.ceil(Ndata / Nbatch))

    if not isinstance(vars, list):
        vars = [vars]

    # Convert names to tensors (if necessary) -----------------
    for i in range(len(vars)):
        if not tf.is_numeric_tensor(vars[i]) and isinstance(vars[i], str):
            vars[i] = tf.get_default_graph().get_tensor_by_name(vars[i])

    # Start batch-inputs --------------------------------------
    y = {}
    for iter in range(Niter):

        sys.stdout.write('\r>> Getting tensors... %d/%d' % (iter + 1, Niter))
        sys.stdout.flush()

        # Get batch -------------------------------------------
        batchidx = np.arange(Nbatch * iter, np.minimum(Nbatch * (iter + 1), Ndata))
        xbatch = x[:, batchidx].T

        # Get tensor data -------------------------------------
        feed_dict = {data_holder: xbatch}
        ybatch = sess.run(vars, feed_dict=feed_dict)

        # Storage
        for tn in range(len(ybatch)):
            # Initialize
            if iter == 0:
                y[tn] = np.zeros([Ndata] + list(ybatch[tn].shape[1:]), dtype=np.float32)
            # Store
            y[tn][batchidx,] = ybatch[tn]

    sys.stdout.write('\r\n')

    return y


# =============================================================
# =============================================================
def calc_accuracy(pred, label, normalize_confmat=True):
    """ Calculate accuracy and confusion matrix
    Args:
        pred: [Ndata x Nlabel]
        label: [Ndata x Nlabel]
    Returns:
        accuracy: accuracy
        conf: confusion matrix
    """

    # print("Calculating accuracy...")

    # Accuracy ------------------------------------------------
    correctflag = pred.reshape(-1) == label.reshape(-1)
    accuracy = np.mean(correctflag)

    # Confusion matrix ----------------------------------------
    conf = confusion_matrix(label[:], pred[:]).astype(np.float32)
    # Normalization
    if normalize_confmat:
        for i in range(conf.shape[0]):
            conf[i, :] = conf[i, :] / np.sum(conf[i, :])

    return accuracy, conf

# =============================================================
# =============================================================
# def correlation(x, y, method='Pearson'):
#     """Evaluate correlation
#      Args:
#          x: data to be sorted
#          y: target data
#      Returns:
#          corr_sort: correlation matrix between x and y (after sorting)
#          sort_idx: sorting index
#          x_sort: x after sorting
#      """
#
#     print("Calculating correlation...")
#
#     x = x.copy()
#     y = y.copy()
#     dim = x.shape[0]
#
#     # Calculate correlation -----------------------------------
#     if method=='Pearson':
#         corr = np.corrcoef(y, x)
#         corr = corr[0:dim,dim:]
#     elif method=='Spearman':
#         corr, pvalue = sp.stats.spearmanr(y.T, x.T)
#         corr = corr[0:dim, dim:]
#
#     # Sort ----------------------------------------------------
#     munk = Munkres()
#     indexes = munk.compute(-np.absolute(corr))
#
#     sort_idx = np.zeros(dim)
#     x_sort = np.zeros(x.shape)
#     for i in range(dim):
#         sort_idx[i] = indexes[i][1]
#         x_sort[i,:] = x[indexes[i][1],:]
#
#     # Re-calculate correlation --------------------------------
#     if method=='Pearson':
#         corr_sort = np.corrcoef(y, x_sort)
#         corr_sort = corr_sort[0:dim,dim:]
#     elif method=='Spearman':
#         corr_sort, pvalue = sp.stats.spearmanr(y.T, x_sort.T)
#         corr_sort = corr_sort[0:dim, dim:]
#
#     return corr_sort, sort_idx, x_sort
