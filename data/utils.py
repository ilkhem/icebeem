import os

import numpy as np


def to_one_hot(x, m=None):
    "batch one hot"
    if type(x) is not list:
        x = [x]
    if m is None:
        ml = []
        for xi in x:
            ml += [xi.max() + 1]
        m = max(ml)
    dtp = x[0].dtype
    xoh = []
    for i, xi in enumerate(x):
        xoh += [np.zeros((xi.size, int(m)), dtype=dtp)]
        xoh[i][np.arange(xi.size), xi.astype(np.int)] = 1
    return xoh


def one_hot_encode(labels, n_labels=10):
    """
    Transforms numeric labels to 1-hot encoded labels. Assumes numeric labels are in the range 0, 1, ..., n_labels-1.
    """

    assert np.min(labels) >= 0 and np.max(labels) < n_labels

    y = np.zeros([labels.size, n_labels]).astype(np.float32)
    y[range(labels.size), labels] = 1

    return y


def single_one_hot_encode(label, n_labels=10):
    """
    Transforms numeric labels to 1-hot encoded labels. Assumes numeric labels are in the range 0, 1, ..., n_labels-1.
    """

    assert label >= 0 and label < n_labels

    y = np.zeros([n_labels]).astype(np.float32)
    y[label] = 1

    return y


def single_one_hot_encode_rev(label, n_labels=10, start_label=0):
    """
    Transforms numeric labels to 1-hot encoded labels. Assumes numeric labels are in the range 0, 1, ..., n_labels-1.
    """
    assert label >= start_label and label < n_labels
    y = np.zeros([n_labels - start_label]).astype(np.float32)
    y[label - start_label] = 1
    return y


mnist_one_hot_transform = lambda label: single_one_hot_encode(label, n_labels=10)
contrastive_one_hot_transform = lambda label: single_one_hot_encode(label, n_labels=2)


def make_dir(dir_name):
    if dir_name[-1] != '/':
        dir_name += '/'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name


def make_file(file_name):
    if not os.path.exists(file_name):
        open(file_name, 'a').close()
    return file_name
