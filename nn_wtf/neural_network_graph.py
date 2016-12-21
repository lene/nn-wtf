from nn_wtf.neural_network_graph_base import NeuralNetworkGraphBase
from nn_wtf.predictor import Predictor
from nn_wtf.trainer import Trainer

import tensorflow as tf

import math

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'


class NeuralNetworkGraph(NeuralNetworkGraphBase):

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        tf.reset_default_graph()

    def output_layer(self):
        assert self.layers, 'called output_layer() before setting up the neural network'
        return self.layers[-1]

    def init_trainer(self, learning_rate=None, optimizer=None, **kwargs):
        assert self.trainer is None, 'init_trainer() called repeatedly'
        self.trainer = Trainer(self, learning_rate=learning_rate, optimizer=optimizer, **kwargs)

    def set_session(self, session=None, **kwargs):
        assert self.trainer is not None, 'need to set the trainer before setting the session'
        if session is None:
            session = _initialize_session()
        self.session = session

    def train(
            self, data_sets, max_steps, precision=None, steps_between_checks=100, run_as_check=None,
            batch_size=None
    ):
        if batch_size is None:
            batch_size = data_sets.train.num_examples

        assert self.session is not None, 'called train() before setting up session'
        assert self.trainer is not None, 'called train() before setting up trainer'
        assert data_sets.train.num_examples % batch_size == 0, \
            'training set size {} not divisible by batch size {}'.format(data_sets.train.num_examples, batch_size)

        self.trainer.train(data_sets, max_steps, precision, steps_between_checks, run_as_check, batch_size)

    def get_predictor(self):
        assert self.session is not None, 'called predictor before setting up a session'
        if self.predictor is None:
            self.predictor = Predictor(self, self.session)
        return self.predictor

    ############################################################################

    def _build_neural_network(self):
        """Builds a neural network with the given layers and output size.

        :return Output tensor with the computed logits.
        """

        assert self.layers == [], 'build_neural_network() has been called before'

        self.layers.append(self.input_placeholder)
        for i in range(1, self.num_hidden_layers+1):
            self.layers.append(
                _add_layer(
                    'layer%04d' % i, self.layer_sizes[i-1], self.layer_sizes[i], self.layers[i-1], tf.nn.relu
                )
            )

        self.layers.append(_add_layer('output', self.layer_sizes[-1], self.output_size, self.layers[-1]))

        return self.output_layer()


def _add_layer(layer_name, in_units_size, out_units_size, input_layer, function=lambda x: x):
    with tf.name_scope(layer_name):
        weights = _initialize_weights(in_units_size, out_units_size)
        biases = _initialize_biases(out_units_size)
        new_layer = function(tf.matmul(input_layer, weights) + biases)

    return new_layer


def _initialize_weights(in_units_size, out_units_size):
    """initialize weights with small amount of noise for symmetry breaking, and to prevent 0 gradients"""
    return tf.Variable(
        tf.truncated_normal([in_units_size, out_units_size], stddev=1.0 / math.sqrt(float(in_units_size))),
        name='weights'
    )

def _initialize_biases(out_units_size):
    return tf.Variable(tf.ones([out_units_size]), name='biases')


def _initialize_session():
    session = tf.Session()
    init = tf.global_variables_initializer()
    session.run(init)
    return session
