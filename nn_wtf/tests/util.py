from nn_wtf.images_labels_data_set import DataSetBase
from nn_wtf.data_sets import DataSets
from nn_wtf.neural_network_graph import NeuralNetworkGraph

import tensorflow as tf

import numpy

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'

MINIMAL_INPUT_SIZE = 2
MINIMAL_LAYER_GEOMETRY = (2, 2)
MINIMAL_OUTPUT_SIZE = 2
MINIMAL_BATCH_SIZE = 2


def create_minimal_input_placeholder():
    return tf.placeholder(tf.float32, shape=(MINIMAL_BATCH_SIZE, MINIMAL_INPUT_SIZE))


def get_project_root_folder():
    import os
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def create_train_data_set():
    train_data = create_vector([0, 0, 1, 1]).reshape(2, 2)
    train_labels = create_vector([0, 1], numpy.int8).reshape(2)
    return DataSetBase(train_data, train_labels)


def create_train_data_sets():
    train_data = create_train_data_set()
    return DataSets(train_data, train_data, train_data)


def create_vector(values, type=numpy.float32):
    return numpy.fromiter(values, numpy.dtype(type))


def train_data_input(value):
    return create_vector([value, value])


def train_neural_network(train_data, graph=None):
    data_sets = DataSets(train_data, train_data, train_data)
    if graph is None:
        graph = NeuralNetworkGraph(train_data.input.shape[0], MINIMAL_LAYER_GEOMETRY, len(train_data.labels))
    init_graph(graph)

    graph.train(
        data_sets=data_sets, steps_between_checks=50, max_steps=1000, batch_size=train_data.num_examples,
        precision=0.99
    )
    return graph


def init_graph(graph, session=None):
    graph.init_trainer()
    graph.set_session(session)
    return graph


def allow_fail(max_times_fail=1, silent=True):
    """Runs a test, if necessary repeatedly, allowing it to fail up to max_times_fail times.

    Usage:

    @allow_fail
    def test: ...
    # allows the test to be repeated once before considering the test failed

    @allow_fail(max_times_fail=2, silent=False)
    def test(): ...
    # allows the test to be repeated twice, printing a message on each failure

    This is useful if a test tests non-deterministic behavior, such as with stochastic
    algorithms. If the tests fails with probability p < 1, allowing it to fail n times
    causes the resulting test to fail with probability p^(n+1) < p.

    In particular it was written to test neural networks which are initialized randomly.

    :param max_times_fail: How often a test may fail before considering the test failed.
    :param silent: If False, prints a message before running the test and on each failure.
    :return: The decorated test method
    """

    def allow_fail_decorator(func):
        """
        :param func: The test allowed to be run repeatedly if it fails.
        :return: The decorated test function
        """
        def run_test_checked(self):
            """ Runs the test, repeating it up to max_times_fail times if it fails.
            :param self: The test suite, presumably an object of type unittest.TestCase
            """
            if not silent:
                print(
                    '\ntrying {}.{}() max. {} times'.format(
                        type(self).__name__, func.__name__, max_times_fail + 1
                    )
                )
            for i in range(max_times_fail):
                try:
                    func(self)
                    return
                except AssertionError:
                    if not silent:
                        print('failed {} times'.format(i+1))
            # run the test unguarded for a last time
            func(self)

        return run_test_checked

    return allow_fail_decorator
