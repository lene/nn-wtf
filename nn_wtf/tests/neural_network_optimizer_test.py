import unittest

from nn_wtf.data_sets import DataSets
from nn_wtf.neural_network_graph import NeuralNetworkGraph
from nn_wtf.parameter_optimizers.brute_force_optimizer import BruteForceOptimizer
from nn_wtf.parameter_optimizers.simulated_annealing_optimizer import SimulatedAnnealingOptimizer
from nn_wtf.tests.util import create_train_data_set

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'
# pylint: disable=missing-docstring


class NeuralNetworkOptimizerTest(unittest.TestCase):

    LAYER_SIZES = ((2,), (2,), (None,))
    DESIRED_PRECISION = 0.5
    MAX_STEPS = 100

    def setUp(self):
        self.train_data = create_train_data_set()
        self.data_sets = DataSets(self.train_data, self.train_data, self.train_data)

    def test_initialize_brute_force_optimizer(self):
        self._create_bf_optimizer()

    def test_brute_force_optimal_network_geometry_runs(self):
        optimizer = self._create_bf_optimizer()
        optimal_layers = optimizer.brute_force_optimal_network_geometry(self.data_sets, self.MAX_STEPS)
        self.assertIsInstance(optimal_layers, tuple)
        self.assertEqual(
            optimal_layers, (self.LAYER_SIZES[0][0], self.LAYER_SIZES[1][0], self.LAYER_SIZES[2][0])
        )

    def test_initialize_simulated_annealing_optimizer(self):
        self._create_sa_optimizer()

    def test_simulated_annealing_optimizer_runs(self):
        optimizer = self._create_sa_optimizer()
        optimizer.best_parameters(self.data_sets, self.MAX_STEPS)

    def test_simulated_annealing_optimizer_more_layers(self):
        optimizer = self._create_sa_optimizer(layer_sizes=((2, 2), (4, 4), (6, 6), (8, 8)))
        optimizer.best_parameters(self.data_sets, self.MAX_STEPS)

    def test_simulated_annealing_optimizer_greater_difference(self):
        optimizer = self._create_sa_optimizer(layer_sizes=((4, 4), (8, 8)), start_size_difference=2)
        optimizer.best_parameters(self.data_sets, self.MAX_STEPS)

    def _create_sa_optimizer(self, layer_sizes=((2,), (2, 2)), start_size_difference=1):
        return SimulatedAnnealingOptimizer(
            NeuralNetworkGraph, end_training_precision=0.5+self.DESIRED_PRECISION/2,
            start_training_precision=self.DESIRED_PRECISION,
            layer_sizes=layer_sizes,
            start_size_difference=start_size_difference,
            input_size=2, output_size=2,
            max_num_before_branching_out=2,
            learning_rate=0.1,
            batch_size=self.train_data.num_examples,
        )

    def _create_bf_optimizer(self):
        return BruteForceOptimizer(
            NeuralNetworkGraph,
            input_size=self.train_data.input.shape[0], output_size=len(self.train_data.labels),
            desired_training_precision=self.DESIRED_PRECISION,
            layer_sizes=self.LAYER_SIZES,
            batch_size=self.train_data.num_examples, verbose=False
        )
