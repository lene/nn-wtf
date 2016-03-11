from nn_wtf.parameter_optimizers.neural_network_optimizer import NeuralNetworkOptimizer

from math import log2
from pprint import pprint

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'


class SimulatedAnnealingOptimizer(NeuralNetworkOptimizer):

    def __init__(
            self, tested_network, start_training_precision, end_training_precision,
            layer_sizes, start_size_difference, input_size, output_size, max_num_before_branching_out=20, learning_rate=None, verbose=False, batch_size=100
    ):
        assert isinstance(input_size, int)
        assert isinstance(output_size, int)
        super().__init__(tested_network, None, None, 0, learning_rate, verbose, batch_size)
        self.start_training_precision = start_training_precision
        self.end_training_precision = end_training_precision
        self.start_layer_sizes = layer_sizes
        self.start_size_difference = start_size_difference
        self.max_num_before_branching_out = max_num_before_branching_out
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.output_size = output_size
        self.current_layer_sizes = list(layer_sizes)
        self.cpu_time_to_step_relevance = 5

    def best_parameters(self, data_sets, max_steps):
        self.batch_size = data_sets.train.num_examples
        timing_results = []
        precisions = self.precisions()
        for iteration_step in range(self.num_iteration_steps()+2):
            self.desired_training_precision = precisions[iteration_step]
            size_difference = self.start_size_difference//2**iteration_step
            self.current_layer_sizes = self.add_neighbors_to_tested_geometries(size_difference, self.current_layer_sizes)
            if self.verbose:
                print(
                    'precision: {:2.6f}, layer sizes: {}'.format(
                        self.desired_training_precision, self.current_layer_sizes
                    )
                )
            timing_results = self.time_all_tested_geometries(data_sets, max_steps)
            timing_results = self.throw_out_runs_that_didnt_reach_required_precision(timing_results)
            timing_results = self.keep_only_best_results(timing_results)
            self.keep_only_best_geometries(timing_results)

        return self.prune_results(data_sets, max_steps, timing_results)

    def prune_results(self, data_sets, max_steps, results):
        while len(self.current_layer_sizes) > 1:
            results = self.time_all_tested_geometries(data_sets, max_steps)
            results = self.keep_only_best_results(results)
            self.keep_only_best_geometries(results)
        return results[0]

    def add_neighbors_to_tested_geometries(self, size_difference, current_layer_sizes):
        neighbors = [
            neighbor
            for old_layer_size in current_layer_sizes
            for neighbor in self.tuple_neighbors(old_layer_size, size_difference)
        ]
        current_layer_sizes = sorted(list(set(neighbors)))
        return current_layer_sizes

    def keep_only_best_geometries(self, results):
        new_layer_sizes = [
            geometry for geometry in self.current_layer_sizes
            if geometry in [result.optimization_parameters.geometry for result in results]
        ]
        self.current_layer_sizes = new_layer_sizes

    def keep_only_best_results(self, results):
        return results[0:self.num_geometries_before_starting_next_generation(results)]

    # TODO
    def throw_out_runs_that_didnt_reach_required_precision(self, results):
        return [result for result in results if result.precision >= self.desired_training_precision]

    def num_geometries_before_starting_next_generation(self, results):
        return max(min(len(results)//2, self.max_num_before_branching_out), 1)

    def tuple_neighbors(self, geometry, difference):
        neighbors = []
        for coordinate in (geometry[0]-difference, geometry[0], geometry[0]+difference):
            if coordinate >= self.output_size and coordinate not in neighbors:
                neighbors.append((coordinate,)+geometry[1:])
                if len(geometry) > 1:
                    neighbors.extend(
                        [
                            (coordinate,)+sub_neighbor
                            for sub_neighbor in self.tuple_neighbors(geometry[1:], difference)
                        ]
                    )
        return neighbors

    def num_iteration_steps(self):
        return int(log2(self.start_size_difference))

    def precisions(self):
        precisions = [self.start_training_precision]
        for i in range(self.num_iteration_steps()):
            next_precision = self.end_training_precision-(self.end_training_precision-precisions[-1])/2.
            precisions.append(next_precision)
        precisions.append(self.end_training_precision)
        return precisions

    def time_all_tested_geometries(self, data_sets, max_steps):
        results = []
        for geometry in self.current_layer_sizes:
            if self.verbose: print('testing', geometry, end='\t', flush=True)
            run_info = self.timed_run_training(
                data_sets,
                NeuralNetworkOptimizer.OptimizationParameters(geometry, self.learning_rate),
                max_steps=max_steps
            )
            if self.verbose: print(run_info)
            run_info.sort_parameter = run_info.cpu_time*self.cpu_time_to_step_relevance+run_info.num_steps
            results.append(run_info)
        results = sorted(results, key=lambda r: r.sort_parameter)
        if self.verbose: pprint(results, width=100)
        return results
