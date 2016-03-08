import time
from math import log2

import tensorflow as tf
from pprint import pprint

from nn_wtf.neural_network_graph_base import NeuralNetworkGraphBase

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'


class NeuralNetworkOptimizer:
    """Attempts to find the best parameters to a neural network to be trained in the fastest way."""

    DEFAULT_LEARNING_RATE = 0.1

    class TimingInfo:

        def __init__(self, cpu_time, wall_time, precision, step, optimization_parameters):
            assert isinstance(optimization_parameters, NeuralNetworkOptimizer.OptimizationParameters)
            self.cpu_time = cpu_time
            self.wall_time = wall_time
            self.precision = precision
            self.num_steps = step
            self.optimization_parameters = optimization_parameters

        def __str__(self):
            return 'CPU: {:7.2f}s Wall: {:7.2f}s Precision: {:5.2f}% Iterations: {:4d} Geometry: {}'.format(
                self.cpu_time, self.wall_time, 100.*self.precision, self.num_steps, str(self.optimization_parameters.geometry)
            )

        def __repr__(self):
            return str(self.__dict__())

        def __dict__(self):
            return {'cpu_time': self.cpu_time, 'step': self.num_steps, 'layers': self.optimization_parameters.geometry}

    class OptimizationParameters:

        def __init__(self, geometry, learning_rate, optimizer=tf.train.GradientDescentOptimizer):
            assert isinstance(geometry, tuple)
            assert learning_rate > 0.
            assert issubclass(optimizer, tf.train.Optimizer)
            self.geometry = geometry
            self.learning_rate = learning_rate
            self.optimizer = optimizer

        def __str__(self):
            return '{} {} {}'.format(self.geometry, self.learning_rate, type(self.optimizer).__name__)

        @classmethod
        def next_parameters(cls, current_parameter):
            pass

    def __init__(
            self, tested_network, input_size, output_size, desired_training_precision,
            learning_rate=None, verbose=False, batch_size=100
    ):
        assert issubclass(tested_network, NeuralNetworkGraphBase)
        assert 0. <= desired_training_precision < 1.
        self.tested_network = tested_network
        self.input_size = input_size
        self.output_size = output_size
        self.desired_training_precision = desired_training_precision
        self.learning_rate = learning_rate if learning_rate else self.DEFAULT_LEARNING_RATE
        self.verbose = verbose
        self.batch_size = batch_size

    def best_parameters(self, data_sets, max_steps):
        raise NotImplementedError

    def timed_run_training(self, data_sets, optimization_parameters, max_steps=10000):
        assert isinstance(optimization_parameters, NeuralNetworkOptimizer.OptimizationParameters)
        graph, cpu, wall = timed_run(self.run_training_once, data_sets, optimization_parameters, max_steps)
        return self.TimingInfo(cpu, wall, graph.trainer.precision(), graph.trainer.num_steps(), optimization_parameters)

    def run_training_once(self, data_sets, optimization_parameters, max_steps):
        assert isinstance(optimization_parameters, NeuralNetworkOptimizer.OptimizationParameters)
        # Tell TensorFlow that the model will be built into the default Graph.
        with tf.Graph().as_default():
            graph = self.tested_network(
                input_size=self.input_size,
                layer_sizes=optimization_parameters.geometry,
                output_size=self.output_size
            )
            graph.init_trainer(learning_rate=optimization_parameters.learning_rate)
            graph.set_session(verbose=self.verbose)
            graph.train(
                data_sets, max_steps,
                precision=self.desired_training_precision, steps_between_checks=50,
                # batch_size=data_sets.train.num_examples
            )
        return graph


class SimulatedAnnealingOptimizer(NeuralNetworkOptimizer):

    def __init__(
            self, tested_network, start_training_precision, desired_training_precision,
            layer_sizes, start_size_difference, learning_rate=None, verbose=False, batch_size=100
    ):
        super().__init__(tested_network, None, None, desired_training_precision, learning_rate, verbose, batch_size)
        self.start_training_precision = start_training_precision
        self.start_layer_sizes = layer_sizes
        self.start_size_difference = start_size_difference
        self.learning_rate = learning_rate
        self.current_layer_sizes = list(layer_sizes)

    def best_parameters(self, data_sets, max_steps):
        self.batch_size = data_sets.train.num_examples
        for iteration_step in range(self.num_iteration_steps()+2):
            precision = self.precisions()[iteration_step]
            size_difference = self.start_size_difference//2**iteration_step
            print('\nprecision:', precision, 'size_difference', size_difference)
            results = self.time_all_tested_geometries(data_sets, max_steps)
            print('results:')
            print(results)
            results = [results[i] for i in range(max(len(results)//2, 1)-1)]
            print('after slicing:',results)
            print('current layer sizes before:', self.current_layer_sizes)
            for i, geometry in enumerate(self.current_layer_sizes):
                if not geometry in [result.optimization_parameters.geometry for result in results]:
                    del self.current_layer_sizes[i]
            print('current layer sizes after:', self.current_layer_sizes)
            self.current_layer_sizes = sorted(list(set(
                [neighbor
                 for old_layer_size in self.current_layer_sizes
                 for neighbor in self.tuple_neighbors(old_layer_size, size_difference)]
            )))
            print(
                'new layer sizes:',
                self.current_layer_sizes
            )
        print('starting to prune results...')
        while len(self.current_layer_sizes) > 1:
            results = [results[i] for i in range(max(len(results)//2, 1)-1)]
            for i, geometry in enumerate(self.current_layer_sizes):
                if not geometry in [result.optimization_parameters.geometry for result in results]:
                    del self.current_layer_sizes[i]
            print('current layer sizes:', self.current_layer_sizes)
            results = self.time_all_tested_geometries(data_sets, max_steps)
            print('results:')
            print(results)
        return results[0]

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
        difference = self.desired_training_precision-self.start_training_precision
        precisions = [self.start_training_precision]
        for i in range(self.num_iteration_steps()):
            next_precision = self.desired_training_precision-(self.desired_training_precision-precisions[-1])/2.
            precisions.append(next_precision)
        precisions.append(self.desired_training_precision)
        return precisions

    def time_all_tested_geometries(self, data_sets, max_steps):
        self.input_size = data_sets.train.input.shape[0]
        self.output_size = data_sets.train.labels.shape[0]
        results = []
        for geometry in self.current_layer_sizes:
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

def timed_run(function, *args, **kwargs):
    start_cpu_time, start_wall_time = time.process_time(), time.time()
    returned = function(*args, **kwargs)
    return returned, time.process_time()-start_cpu_time, time.time()-start_wall_time
