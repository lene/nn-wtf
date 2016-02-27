from nn_wtf.images_labels_data_set import DataSetBase

import tensorflow as tf

import numpy

__author__ = 'Lene Preuss <lp@sinnwerkstatt.com>'

MINIMAL_INPUT_SIZE = 2
MINIMAL_LAYER_GEOMETRY = (2, 2)
MINIMAL_OUTPUT_SIZE = 2
MINIMAL_BATCH_SIZE = 2


def create_minimal_input_placeholder():
    return tf.placeholder(tf.float32, shape=(MINIMAL_BATCH_SIZE, MINIMAL_INPUT_SIZE))


def get_project_root_folder():
    import os
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def create_train_data_set():
    train_data = numpy.fromiter([0, 0, 1, 1], numpy.dtype(numpy.float32)).reshape(2, 2)
    train_labels = numpy.fromiter([0, 1], numpy.dtype(numpy.int8)).reshape(2)
    return DataSetBase(train_data, train_labels)


def train_data_input(value):
    return numpy.fromiter([value, value], numpy.dtype(numpy.float32))


