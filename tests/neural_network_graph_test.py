from neural_network_graph import NeuralNetworkGraph

import unittest

__author__ = 'Lene Preuss <lp@sinnwerkstatt.com>'


class NeuralNetworkGraphTest(unittest.TestCase):

    def test_init_runs(self):
        NeuralNetworkGraph(2, (2,2), 2)

    def test_init_fails_on_bad_layer_sizes(self):
        with self.assertRaises(TypeError):
            NeuralNetworkGraph(2, 2, 2)

    def test_init_fails_if_last_layer_smaller_than_output_size(self):
        with self.assertRaises(ValueError):
            NeuralNetworkGraph(2, (2, 1), 2)
