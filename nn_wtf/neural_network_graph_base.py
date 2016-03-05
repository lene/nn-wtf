from nn_wtf.data_set_base import DataSetBase

import tensorflow as tf

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'


class NeuralNetworkGraphBase:

    def __init__(self, input_size, layer_sizes, output_size):
        """Initialize a neural network given its geometry.

        :param input_size: number of input channels
        :param layer_sizes: tuple of sizes of the neural network hidden layers
        :param output_size: number of output classes
        """
        self._setup_geometry(input_size, layer_sizes, output_size)
        self.predictor = None
        self.trainer = None
        self.session = None
        self.layers = []
        self.input_placeholder = tf.placeholder(tf.float32, shape=(None, self.input_size), name='input')
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None,), name='labels')
        self._build_neural_network()

    def output_layer(self):
        raise NotImplementedError

    def fill_feed_dict(self, data_set, batch_size):
        """Fills the feed_dict for training the given step.

        A feed_dict takes the form of:
        feed_dict = {
            <placeholder>: <tensor of values to be passed for placeholder>,
              ....
        }

        :param data_set: The set of images and labels
        :param batch_size: Number of data sets to work on as one batch
        :return The feed dictionary mapping from placeholders to values.
        """
        # Create the feed_dict for the placeholders filled with the next `batch size ` examples.
        assert isinstance(data_set, DataSetBase)
        input_feed, labels_feed = data_set.next_batch(batch_size)
        feed_dict = {
            self.input_placeholder: input_feed,
            self.labels_placeholder: labels_feed,
        }
        return feed_dict

    ############################################################################

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


