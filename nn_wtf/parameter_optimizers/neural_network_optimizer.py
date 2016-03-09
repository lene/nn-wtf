import time

import tensorflow as tf

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


def timed_run(function, *args, **kwargs):
    start_cpu_time, start_wall_time = time.process_time(), time.time()
    returned = function(*args, **kwargs)
    return returned, time.process_time()-start_cpu_time, time.time()-start_wall_time
