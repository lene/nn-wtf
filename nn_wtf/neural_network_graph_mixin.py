import tensorflow as tf

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'

DEFAULT_TRAIN_DIR = '.nn_wtf-data'


class NeuralNetworkGraphMixin:
    def __init__(self, session, train_dir=DEFAULT_TRAIN_DIR):
        assert isinstance(session, tf.Session), 'session must be set when initializing '+type(self).__name__
        self.session = session
        self.train_dir = ensure_is_dir(train_dir)


def ensure_is_dir(train_dir_string):
    if not train_dir_string[-1] == '/':
        train_dir_string += '/'
    return train_dir_string

