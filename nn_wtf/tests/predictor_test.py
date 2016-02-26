from nn_wtf.neural_network_graph import NeuralNetworkGraph

from .util import MINIMAL_INPUT_SIZE, MINIMAL_OUTPUT_SIZE, MINIMAL_LAYER_GEOMETRY, MINIMAL_BATCH_SIZE
from .util import create_minimal_input_placeholder

import tensorflow as tf

import unittest

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'

class NeuralNetworkGraphTest(unittest.TestCase):
