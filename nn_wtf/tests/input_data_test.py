from nn_wtf.images_labels_data_set import ImagesLabelsDataSet
from nn_wtf.input_data import read_images_from_file
from nn_wtf.mnist_data_sets import MNISTDataSets

from nn_wtf.mnist_graph import MNISTGraph

from .util import get_project_root_folder

import numpy

import unittest

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'
# pylint: disable=missing-docstring


class InputDataTest(unittest.TestCase):

    def setUp(self):
        self.data = MNISTDataSets.read_one_image_from_file(get_project_root_folder()+'/nn_wtf/data/0.raw')

    def test_read_one_image_from_file(self):
        self.assertIsInstance(self.data, numpy.ndarray)
        self._check_is_one_mnist_image(self.data)

    def test_image_labels_data_set_from_image(self):
        labels = numpy.fromiter([0], dtype=numpy.uint8)
        data_set = ImagesLabelsDataSet(self.data, labels)

    def test_read_images_from_file_one(self):
        data = read_images_from_file(
            get_project_root_folder()+'/nn_wtf/data/0.raw',
            MNISTGraph.IMAGE_SIZE, MNISTGraph.IMAGE_SIZE, 1
        )
        self._check_is_one_mnist_image(data)

    def test_read_images_from_file_two(self):
        data = read_images_from_file(
            get_project_root_folder()+'/nn_wtf/data/7_2.raw',
            MNISTGraph.IMAGE_SIZE, MNISTGraph.IMAGE_SIZE, 2
        )
        self._check_is_n_mnist_images(2, data)

    def test_read_images_from_file_fails_if_file_too_short(self):
        with self.assertRaises(ValueError):
            read_images_from_file(
                get_project_root_folder()+'/nn_wtf/data/7_2.raw',
                MNISTGraph.IMAGE_SIZE, MNISTGraph.IMAGE_SIZE, 3
            )

    def test_read_images_from_file_two_using_mnist_data_sets(self):
        data = MNISTDataSets.read_images_from_file(
            get_project_root_folder()+'/nn_wtf/data/7_2.raw', 2
        )
        self._check_is_n_mnist_images(2, data)

    def test_read_images_from_file_using_mnist_data_sets_fails_if_file_too_short(self):
        with self.assertRaises(ValueError):
            MNISTDataSets.read_images_from_file(
                get_project_root_folder()+'/nn_wtf/data/7_2.raw', 3
            )

    def _check_is_one_mnist_image(self, data):
        self._check_is_n_mnist_images(1, data)

    def _check_is_n_mnist_images(self, n, data):
        self.assertEqual(4, len(data.shape))
        self.assertEqual(n, data.shape[0])
        self.assertEqual(MNISTGraph.IMAGE_SIZE, data.shape[1])
        self.assertEqual(MNISTGraph.IMAGE_SIZE, data.shape[2])
        self.assertEqual(1, data.shape[3])

