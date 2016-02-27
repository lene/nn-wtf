import time
import pprint
import tensorflow as tf

from nn_wtf.neural_network_graph import NeuralNetworkGraph

__author__ = 'Lene Preuss <lp@sinnwerkstatt.com>'


class NeuralNetworkOptimizer:
    """Attempts to find the best parameters to a neural network to be trained in the fastest way."""

    DEFAULT_LEARNING_RATE = 0.1
    DEFAULT_LAYER_SIZES = (
        (32, 48, 64, 80),
        (32, 48, 64, 80),
        # (32, 48, 64, 80, 96, 128),
        # (32, 48, 64, 80, 96, 128),
        (None, 16, 32, 48)
    )

    class TimingInfo:

        def __init__(self, cpu_time, wall_time, precision, step, layers):
            self.cpu_time = cpu_time
            self.wall_time = wall_time
            self.precision = precision
            self.step = step
            self.layers = layers

        def __str__(self):
            return 'CPU: {:7.2f}s Wall: {:7.2f}s Precision: {:5.2f}% Iterations: {:4d} Geometry: {}'.format(
                self.cpu_time, self.wall_time, 100.*self.precision, self.step, str(self.layers)
            )

        def __repr__(self):
            return str(self.__dict__())

        def __dict__(self):
            return {'cpu_time': self.cpu_time, 'step': self.step, 'layers': self.layers}

    def __init__(
            self, tested_network, input_size, output_size, training_precision,
            layer_sizes=None, learning_rate=None, verbose=False, batch_size=100
    ):
        assert issubclass(tested_network, NeuralNetworkGraph)
        self.tested_network = tested_network
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.verbose = verbose
        self.learning_rate = learning_rate if learning_rate else self.DEFAULT_LEARNING_RATE
        self.training_precision = training_precision
        self.layer_sizes = self.DEFAULT_LAYER_SIZES if layer_sizes is None else layer_sizes

    def brute_force_optimal_network_geometry(self, data_sets, max_steps):
        results = self.time_all_tested_geometries(data_sets, max_steps)
        return results[0].layers

    def time_all_tested_geometries(self, data_sets, max_steps):
        results = []
        for geometry in self.get_network_geometries():
            run_info = self.timed_run_training(
                data_sets, geometry, max_steps=max_steps
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

    def timed_run_training(self, data_sets, geometry, max_steps=10000):
        graph, cpu, wall = timed_run(self.run_training_once, data_sets, geometry, max_steps)
        return self.TimingInfo(cpu, wall, graph.trainer.precision, graph.trainer.step, geometry)

    def run_training_once(self, data_sets, geometry, max_steps):
        # Tell TensorFlow that the model will be built into the default Graph.
        with tf.Graph().as_default():
            graph = self.tested_network(
                input_size=self.input_size,
                layer_sizes=geometry,
                output_size=self.output_size,
                learning_rate=self.learning_rate
            )
            graph.set_session()
            graph.train(
                data_sets, max_steps,
                precision=self.training_precision, steps_between_checks=50, batch_size=self.batch_size
            )
        return graph


def timed_run(function, *args, **kwargs):
    start_cpu_time, start_wall_time = time.process_time(), time.time()
    returned = function(*args, **kwargs)
    return returned, time.process_time()-start_cpu_time, time.time()-start_wall_time
