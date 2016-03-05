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

    def _build_neural_network(self):
        raise NotImplementedError

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


