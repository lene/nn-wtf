from nn_wtf.predictor import Predictor
from nn_wtf.trainer import Trainer

import tensorflow as tf

import math

__author__ = 'Lene Preuss <lp@sinnwerkstatt.com>'

# todo, obvs :-)
CHANGE_THIS_LEARNING_RATE = 0.1


class NeuralNetworkGraph:

    def __init__(self, input_size, layer_sizes, output_size, learning_rate=CHANGE_THIS_LEARNING_RATE):
        """Initialize a neural network given its geometry.

        :param input_size: number of input channels
        :param layer_sizes: tuple of sizes of the neural network hidden layers
        :param output_size: number of output classes
        :param learning_rate: learning rate for gradient descent optimizer
        """
        self._setup_geometry(input_size, layer_sizes, output_size)
        self.learning_rate = learning_rate
        self.predictor = None
        self.trainer = None
        self.layers = []
        self.input_placeholder = tf.placeholder(tf.float32, shape=(None, self.input_size), name='input')
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None,), name='labels')
        self._build_neural_network()

    def output_layer(self):
        return self.layers[-1]

    def set_session(self, session=None):
        if self.trainer is None:
            self.trainer = Trainer(self, learning_rate=self.learning_rate)

        if session is None:
            session = _initialize_session()

        self.session = session

    def train(
            self, data_sets, max_steps, precision=None, steps_between_checks=100, run_as_check=None,
            batch_size=1000
    ):
        assert self.session is not None, 'called train() before setting up session'
        assert self.trainer is not None, 'called train() before setting up trainer'

        self.trainer.train(data_sets, max_steps, precision, steps_between_checks, run_as_check, batch_size)

    def fill_feed_dict(self, data_set, batch_size):
        """Fills the feed_dict for training the given step.

        A feed_dict takes the form of:
        feed_dict = {
            <placeholder>: <tensor of values to be passed for placeholder>,
              ....
        }

        :param data_set: The set of images and labels, from input_data.read_data_sets()
        :return The feed dictionary mapping from placeholders to values.
        """
        # Create the feed_dict for the placeholders filled with the next `batch size ` examples.
        input_feed, labels_feed = data_set.next_batch(batch_size)
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed,
        }
        return feed_dict

    def get_predictor(self):
        assert self.session is not None, 'called predictor before setting up a session'
        if self.predictor is None:
            self.predictor = Predictor(self, self.session)
        return self.predictor

    def _setup_geometry(self, input_size, layer_sizes, output_size):
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.layer_sizes = self._set_layer_sizes(layer_sizes)
        self.num_hidden_layers = len(self.layer_sizes) - 1

    def _set_layer_sizes(self, layer_sizes):
        layer_sizes = tuple(filter(None, layer_sizes))
        if layer_sizes[-1] < self.output_size:
            raise ValueError('Last layer size must be greater or equal output size')
        return (self.input_size,) + layer_sizes

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
    init = tf.initialize_all_variables()
    session.run(init)
    return session
