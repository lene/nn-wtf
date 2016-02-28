from nn_wtf.neural_network_graph import NeuralNetworkGraph

from .util import MINIMAL_INPUT_SIZE, MINIMAL_OUTPUT_SIZE, MINIMAL_LAYER_GEOMETRY

import tensorflow as tf

import unittest

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'
# pylint: disable=missing-docstring


class NeuralNetworkGraphTest(unittest.TestCase):

    def test_init_runs(self):
        NeuralNetworkGraph(MINIMAL_INPUT_SIZE, MINIMAL_LAYER_GEOMETRY, MINIMAL_OUTPUT_SIZE)

    def test_init_fails_on_bad_layer_sizes(self):
        with self.assertRaises(TypeError):
            NeuralNetworkGraph(2, 2, 2)

    def test_init_fails_if_last_layer_smaller_than_output_size(self):
        with self.assertRaises(ValueError):
            NeuralNetworkGraph(2, (2, 1), 2)

    def test_build_neural_network_runs_only_once(self):
        graph = self._create_minimal_graph()
        with self.assertRaises(AssertionError):
            graph._build_neural_network()

    def test_build_neural_network_output(self):
        graph = self._create_minimal_graph()

        self.assertIsInstance(graph.output_layer(), tf.Tensor)
        self.assertEqual(2, graph.output_layer().get_shape().ndims)
        self.assertEqual(MINIMAL_OUTPUT_SIZE, int(graph.output_layer().get_shape()[1]))
        self.assertEqual(len(graph.layers)-2, graph.num_hidden_layers)
        self.assertEqual(len(MINIMAL_LAYER_GEOMETRY), graph.num_hidden_layers)

    def test_build_neural_network_output_with_three_layers(self):
        self._check_num_hidden_layers_for_input_is((4, 3, 2), 3)

    def test_build_neural_network_output_with_last_layer_none(self):
        self._check_num_hidden_layers_for_input_is((4, 3, None), 2)

    def test_build_neural_network_output_with_middle_layer_none(self):
        self._check_num_hidden_layers_for_input_is((4, None, 2), 2)

    def _check_num_hidden_layers_for_input_is(self, definition, expected_size):
        graph = NeuralNetworkGraph(MINIMAL_INPUT_SIZE, definition, MINIMAL_OUTPUT_SIZE)
        self.assertEqual(expected_size, graph.num_hidden_layers)

    def _create_minimal_graph(self):
        return NeuralNetworkGraph(MINIMAL_INPUT_SIZE, MINIMAL_LAYER_GEOMETRY, MINIMAL_OUTPUT_SIZE)

