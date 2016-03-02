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

    # Because reading from URLs is slow and unreliable, this is usually turned off.
    TEST_READ_URLS = False

    SAMPLE_MNIST_FILE = get_project_root_folder() + '/nn_wtf/data/handwritten_test_data/0.raw'
    SAMPLE_MNIST_FILE_WITH_TWO_IMAGES = get_project_root_folder()+'/nn_wtf/data/7_2.raw'
    SAMPLE_MNIST_URL_1 = 'http://github.com/lene/nn-wtf/blob/master/nn_wtf/data/7_from_test_set.raw?raw=true'
    SAMPLE_MNIST_URL_2 = 'http://github.com/lene/nn-wtf/blob/master/nn_wtf/data/1_from_test_set.raw?raw=true'

    def setUp(self):
        self.data = MNISTDataSets.read_one_image_from_file(self.SAMPLE_MNIST_FILE)

    def test_read_one_image_from_file(self):
        self.assertIsInstance(self.data, numpy.ndarray)
        self._check_is_one_mnist_image(self.data)

    def test_image_labels_data_set_from_image(self):
        labels = numpy.fromiter([0], dtype=numpy.uint8)
        data_set = ImagesLabelsDataSet(self.data, labels)

    def test_read_images_from_file_one(self):
        data = read_images_from_file(
            self.SAMPLE_MNIST_FILE,
            MNISTGraph.IMAGE_SIZE, MNISTGraph.IMAGE_SIZE, 1
        )
        self._check_is_one_mnist_image(data)

    def test_read_images_from_file_two(self):
        data = read_images_from_file(
            self.SAMPLE_MNIST_FILE_WITH_TWO_IMAGES,
            MNISTGraph.IMAGE_SIZE, MNISTGraph.IMAGE_SIZE, 2
        )
        self._check_is_n_mnist_images(2, data)

    def test_read_images_from_file_fails_if_file_too_short(self):
        with self.assertRaises(ValueError):
            read_images_from_file(
                self.SAMPLE_MNIST_FILE_WITH_TWO_IMAGES,
                MNISTGraph.IMAGE_SIZE, MNISTGraph.IMAGE_SIZE, 3
            )

    def test_read_images_from_file_two_using_mnist_data_sets(self):
        data = MNISTDataSets.read_images_from_file(
            self.SAMPLE_MNIST_FILE_WITH_TWO_IMAGES, 2
        )
        self._check_is_n_mnist_images(2, data)

    def test_read_images_from_file_using_mnist_data_sets_fails_if_file_too_short(self):
        with self.assertRaises(ValueError):
            MNISTDataSets.read_images_from_file(
                self.SAMPLE_MNIST_FILE_WITH_TWO_IMAGES, 3
            )

    def test_read_images_from_two_files_using_mnist_data_sets(self):
        data = MNISTDataSets.read_images_from_files(
            get_project_root_folder()+'/nn_wtf/data/7_from_test_set.raw',
            get_project_root_folder()+'/nn_wtf/data/2_from_test_set.raw'
        )
        self._check_is_n_mnist_images(2, data)

    def test_read_images_from_three_files_using_mnist_data_sets(self):
        data = MNISTDataSets.read_images_from_files(
            get_project_root_folder()+'/nn_wtf/data/7_from_test_set.raw',
            get_project_root_folder()+'/nn_wtf/data/2_from_test_set.raw',
            get_project_root_folder()+'/nn_wtf/data/1_from_test_set.raw'
        )
        self._check_is_n_mnist_images(3, data)

    def test_read_one_image_from_url(self):
        if not self.TEST_READ_URLS:
            self.skipTest('Testing reading from URLs disabled')
        data = MNISTDataSets.read_one_image_from_url(self.SAMPLE_MNIST_URL_1)
        self._check_is_one_mnist_image(data)

    def test_read_images_from_two_urls_using_mnist_data_sets(self):
        if not self.TEST_READ_URLS:
            self.skipTest('Testing reading from URLs disabled')
        data = MNISTDataSets.read_images_from_urls(
            self.SAMPLE_MNIST_URL_1, self.SAMPLE_MNIST_URL_2
        )
        self._check_is_n_mnist_images(2, data)

    def _check_is_one_mnist_image(self, data):
        self._check_is_n_mnist_images(1, data)

    def _check_is_n_mnist_images(self, n, data):
        self.assertEqual(4, len(data.shape))
        self.assertEqual(n, data.shape[0])
        self.assertEqual(MNISTGraph.IMAGE_SIZE, data.shape[1])
        self.assertEqual(MNISTGraph.IMAGE_SIZE, data.shape[2])
        self.assertEqual(1, data.shape[3])

