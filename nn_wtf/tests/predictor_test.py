from nn_wtf.data_sets import DataSets
from nn_wtf.neural_network_graph import NeuralNetworkGraph
from nn_wtf.tests.util import MINIMAL_LAYER_GEOMETRY, create_train_data_set, train_data_input, \
    create_vector, allow_fail

import unittest
import numpy

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'
# pylint: disable=missing-docstring


class PredictorTest(unittest.TestCase):

    @allow_fail(max_times_fail=1)
    def test_all_prediction_functions_at_once_to_save_computing_time(self):
        """Training takes time, if I run tests separately I have to train for each test."""

        graph = train_neural_network(create_train_data_set())

        self.assertEqual(0, graph.get_predictor().predict(train_data_input(0)))
        self.assertEqual(1, graph.get_predictor().predict(train_data_input(1)))

        probabilities_for_0 = graph.get_predictor().prediction_probabilities(train_data_input(0))
        self.assertGreater(probabilities_for_0[0], probabilities_for_0[1])

        probabilities_for_1 = graph.get_predictor().prediction_probabilities(train_data_input(1))
        self.assertGreater(probabilities_for_1[1], probabilities_for_1[0])

        predictions = graph.get_predictor().predict_multiple(train_data_0_1(), 2)
        self.assertEqual([0, 1], predictions)

        predictions = graph.get_predictor().predict_multiple(train_data_1_0(), 2)
        self.assertEqual([1, 0], predictions)

        predictions = graph.get_predictor().predict_multiple(train_data_0_1_0(), 3)
        self.assertEqual([0, 1, 0], predictions)


def train_neural_network(train_data):
    data_sets = DataSets(train_data, train_data, train_data)
    graph = NeuralNetworkGraph(train_data.input.shape[0], MINIMAL_LAYER_GEOMETRY, len(train_data.labels))
    graph.set_session()

    graph.train(
        data_sets=data_sets, steps_between_checks=50, max_steps=1000, batch_size=train_data.num_examples,
        precision=0.99
    )
    return graph


def train_data_0_1():
    return create_vector([0, 0, 1, 1])


def train_data_1_0():
    return create_vector([1, 1, 0, 0])


def train_data_0_1_0():
    return create_vector([0, 0, 1, 1, 0, 0])
