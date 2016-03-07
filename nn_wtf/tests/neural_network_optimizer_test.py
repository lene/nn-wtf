from nn_wtf.data_sets import DataSets
from nn_wtf.neural_network_optimizer import BruteForceOptimizer
from nn_wtf.neural_network_graph import NeuralNetworkGraph

from nn_wtf.tests.util import create_train_data_set

import unittest

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'
# pylint: disable=missing-docstring


class NeuralNetworkOptimizerTest(unittest.TestCase):

    LAYER_SIZES = ((2,), (2,), (None,))
    DESIRED_PRECISION = 0.5
    MAX_STEPS = 100

    def setUp(self):
        self.train_data = create_train_data_set()
        self.data_sets = DataSets(self.train_data, self.train_data, self.train_data)

    def test_initialize_optimizer(self):
        self._create_optimizer(BruteForceOptimizer)

    def test_brute_force_optimal_network_geometry_runs(self):
        optimizer = self._create_optimizer(BruteForceOptimizer)
        optimal_layers = optimizer.brute_force_optimal_network_geometry(self.data_sets, self.MAX_STEPS)
        self.assertIsInstance(optimal_layers, tuple)
        self.assertEqual(
            optimal_layers, (self.LAYER_SIZES[0][0], self.LAYER_SIZES[1][0], self.LAYER_SIZES[2][0])
        )

    def _create_optimizer(self, optimizer_class):
        return optimizer_class(
            NeuralNetworkGraph,
            self.train_data.input.shape[0], len(self.train_data.labels),
            self.DESIRED_PRECISION,
            batch_size=self.train_data.num_examples,
            layer_sizes=self.LAYER_SIZES
        )
