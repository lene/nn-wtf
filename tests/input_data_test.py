from images_labels_data_set import ImagesLabelsDataSet
from input_data import read_one_image_from_file

from mnist_graph import IMAGE_SIZE
import numpy

import unittest

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'


class InputDataTest(unittest.TestCase):

    def setUp(self):
        self.data = read_one_image_from_file('data/0.raw')

    def test_read_one_image_from_file(self):
        self.assertIsInstance(self.data, numpy.ndarray)
        self.assertEqual(4, len(self.data.shape))
        self.assertEqual(1, self.data.shape[0])
        self.assertEqual(IMAGE_SIZE, self.data.shape[1])
        self.assertEqual(IMAGE_SIZE, self.data.shape[2])
        self.assertEqual(1, self.data.shape[3])

    def test_image_labels_data_set_from_image(self):
        labels = numpy.fromiter([0], dtype=numpy.uint8)
        data_set = ImagesLabelsDataSet(self.data, labels)
