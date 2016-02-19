from data_set import DataSet
from .util import create_minimal_input_placeholder

import tensorflow as tf
import numpy

import unittest

__author__ = 'Lene Preuss <lp@sinnwerkstatt.com>'

NUM_TRAINING_SAMPLES = 20
IMAGE_WIDTH = 10
IMAGE_HEIGHT = 10

class DataSetTest(unittest.TestCase):

    def test_init_without_fake_data_runs(self):
        images = self.create_empty_image_data()
        labels = self.create_empty_label_data()
        DataSet(images, labels)

    def test_init_with_fake_data_runs(self):
        images = create_minimal_input_placeholder()
        DataSet(images, images, fake_data=True)

    def create_empty_image_data(self):
        buf = [0] * NUM_TRAINING_SAMPLES * IMAGE_WIDTH * IMAGE_HEIGHT
        data = numpy.fromiter(buf, dtype=numpy.uint8)
        return data.reshape(NUM_TRAINING_SAMPLES, IMAGE_WIDTH, IMAGE_HEIGHT, 1)

    def create_empty_label_data(self):
        buf = [0]*NUM_TRAINING_SAMPLES
        return numpy.fromiter(buf, dtype=numpy.uint8)
