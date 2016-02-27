from nn_wtf.images_labels_data_set import ImagesLabelsDataSet
from nn_wtf.input_data import read_one_image_from_file

from nn_wtf.mnist_graph import MNISTGraph

from .util import get_project_root_folder

import numpy

import unittest

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'
# pylint: disable=missing-docstring


class InputDataTest(unittest.TestCase):

    def setUp(self):
        self.data = read_one_image_from_file(get_project_root_folder()+'/nn_wtf/data/0.raw')

    def test_read_one_image_from_file(self):
        self.assertIsInstance(self.data, numpy.ndarray)
        self.assertEqual(4, len(self.data.shape))
        self.assertEqual(1, self.data.shape[0])
        self.assertEqual(MNISTGraph.IMAGE_SIZE, self.data.shape[1])
        self.assertEqual(MNISTGraph.IMAGE_SIZE, self.data.shape[2])
        self.assertEqual(1, self.data.shape[3])

    def test_image_labels_data_set_from_image(self):
        labels = numpy.fromiter([0], dtype=numpy.uint8)
        data_set = ImagesLabelsDataSet(self.data, labels)

