from nn_wtf.neural_network_graph import NeuralNetworkGraph

from .util import MINIMAL_INPUT_SIZE, MINIMAL_OUTPUT_SIZE, MINIMAL_LAYER_GEOMETRY, MINIMAL_BATCH_SIZE
from .util import create_minimal_input_placeholder

import tensorflow as tf

import unittest

__author__ = 'Lene Preuss <lp@sinnwerkstatt.com>'



class NeuralNetworkGraphTest(unittest.TestCase):

    def test_init_runs(self):
        NeuralNetworkGraph(MINIMAL_INPUT_SIZE, MINIMAL_LAYER_GEOMETRY, MINIMAL_OUTPUT_SIZE)

    def test_init_fails_on_bad_layer_sizes(self):
        with self.assertRaises(TypeError):
            NeuralNetworkGraph(2, 2, 2)

    def test_init_fails_if_last_layer_smaller_than_output_size(self):
        with self.assertRaises(ValueError):
            NeuralNetworkGraph(2, (2, 1), 2)

    def test_build_neural_network_runs(self):
        graph = self._create_minimal_graph()
        input_placeholder = create_minimal_input_placeholder()

        graph.build_neural_network(input_placeholder)

    def test_build_neural_network_runs_only_once(self):
        graph = self._create_minimal_graph()
        input_placeholder = create_minimal_input_placeholder()

        graph.build_neural_network(input_placeholder)

        with self.assertRaises(AssertionError):
            graph.build_neural_network(input_placeholder)

    def test_build_neural_network_fails_on_bad_input_size(self):
        graph = self._create_minimal_graph()
        input_placeholder = tf.placeholder(tf.float32, shape=(MINIMAL_BATCH_SIZE, MINIMAL_INPUT_SIZE+1))

        with self.assertRaises(AssertionError):
            graph.build_neural_network(input_placeholder)

    def test_build_neural_network_output(self):
        graph = self._create_minimal_graph()
        input_placeholder = create_minimal_input_placeholder()

        output = graph.build_neural_network(input_placeholder)

        self.assertIsInstance(output, tf.Tensor)
        self.assertEqual(2, output.get_shape().ndims)
        self.assertEqual(MINIMAL_OUTPUT_SIZE, int(output.get_shape()[1]))

    def _create_minimal_graph(self):
        return NeuralNetworkGraph(MINIMAL_INPUT_SIZE, MINIMAL_LAYER_GEOMETRY, MINIMAL_OUTPUT_SIZE)


