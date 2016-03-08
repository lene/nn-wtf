from nn_wtf.neural_network_optimizer import NeuralNetworkOptimizer

import pprint

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'


class BruteForceOptimizer(NeuralNetworkOptimizer):

    DEFAULT_LAYER_SIZES = (
        (32, 48, 64),  # (32, 48, 64, 80, 96, 128),
        (32, 48, 64, 80, 96, 128),
        (None, 16, 32, 48)
    )

    def __init__(
            self, tested_network, input_size, output_size, desired_training_precision,
            layer_sizes=None, learning_rate=None, verbose=False, batch_size=100
    ):
        super().__init__(tested_network, input_size, output_size, desired_training_precision, verbose, batch_size)
        self.learning_rate = learning_rate if learning_rate else self.DEFAULT_LEARNING_RATE
        self.layer_sizes = self.DEFAULT_LAYER_SIZES if layer_sizes is None else layer_sizes

    def best_parameters(self, data_sets, max_steps):
        results = self.time_all_tested_geometries(data_sets, max_steps)
        return results[0].optimization_parameters

    def brute_force_optimal_network_geometry(self, data_sets, max_steps):
        return self.best_parameters(data_sets, max_steps).geometry

    def time_all_tested_geometries(self, data_sets, max_steps):
        results = []
        for geometry in self.get_network_geometries():
            run_info = self.timed_run_training(
                data_sets,
                NeuralNetworkOptimizer.OptimizationParameters(geometry, self.learning_rate),
                max_steps=max_steps
            )
            if self.verbose: print(run_info)
            results.append(run_info)
        results = sorted(results, key=lambda r: r.cpu_time)
        if self.verbose: pprint.pprint(results, width=100)
        return results

    def get_network_geometries(self):
        return ((l1, l2, l3)
                for l1 in self.layer_sizes[0]
                for l2 in self.layer_sizes[1] if l2 <= l1
                for l3 in self.layer_sizes[2] if l3 is None or l3 <= l2)

    def brute_force_optimize_learning_rate(self):
        raise NotImplementedError()

