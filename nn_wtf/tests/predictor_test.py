from nn_wtf.predictor import Predictor
from nn_wtf.neural_network_graph import NeuralNetworkGraph
from nn_wtf.tests.util import create_minimal_input_placeholder

from .util import MINIMAL_INPUT_SIZE, MINIMAL_OUTPUT_SIZE, MINIMAL_LAYER_GEOMETRY, MINIMAL_BATCH_SIZE

import tensorflow as tf

import unittest

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'


class PredictorTest(unittest.TestCase):

    def test_simple_case(self):
        graph = NeuralNetworkGraph(MINIMAL_INPUT_SIZE, MINIMAL_LAYER_GEOMETRY, MINIMAL_OUTPUT_SIZE)
        output = graph.build_neural_network()
        train_data = [[0, 0], [1, 1]]
        train_labels = [0, 1]

