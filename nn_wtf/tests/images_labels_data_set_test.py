from nn_wtf.images_labels_data_set import ImagesLabelsDataSet
from .util import create_minimal_input_placeholder

import numpy

import unittest

__author__ = 'Lene Preuss <lp@sinnwerkstatt.com>'
# pylint: disable=missing-docstring

NUM_TRAINING_SAMPLES = 20
IMAGE_WIDTH = 10
IMAGE_HEIGHT = 10
BATCH_SIZE = 5


class ImagesLabelsDataSetTest(unittest.TestCase):

    def test_init_without_fake_data_runs(self):
        _create_empty_data_set()

    def test_init_with_different_label_size_fails(self):
        images = create_empty_image_data()
        labels = create_empty_label_dataof_size(NUM_TRAINING_SAMPLES+1)
        with self.assertRaises(AssertionError):
            ImagesLabelsDataSet(images, labels)

    def test_init_with_wrong_label_shape_fails(self):
        images = create_empty_image_data()
        with self.assertRaises(AssertionError):
            ImagesLabelsDataSet(images, images)

    def test_next_batch_returns_correct_data_format(self):
        data_set = _create_empty_data_set()
        images, labels = data_set.next_batch(BATCH_SIZE)
        self.assertIsInstance(images, numpy.ndarray)
        self.assertEqual(2, len(images.shape))
        self.assertEqual(BATCH_SIZE, images.shape[0])
        self.assertEqual(IMAGE_WIDTH*IMAGE_HEIGHT, images.shape[1])
        self.assertIsInstance(labels, numpy.ndarray)
        self.assertEqual(1, len(labels.shape))
        self.assertEqual(BATCH_SIZE, labels.shape[0])

    def test_next_batch_runs_repeatedly(self):
        data_set = _create_empty_data_set()
        batch_size = NUM_TRAINING_SAMPLES//2
        _, _ = data_set.next_batch(batch_size)
        _, _ = data_set.next_batch(batch_size)

    def test_next_batch_sets_epochs_completed(self):
        data_set = _create_empty_data_set()
        batch_size = NUM_TRAINING_SAMPLES//2
        self.assertEqual(0, data_set.epochs_completed)
        _, _ = data_set.next_batch(batch_size)
        self.assertEqual(0, data_set.epochs_completed)
        _, _ = data_set.next_batch(batch_size)
        self.assertEqual(0, data_set.epochs_completed)
        _, _ = data_set.next_batch(batch_size)
        self.assertEqual(1, data_set.epochs_completed)


def _create_empty_data_set():
    images = create_empty_image_data()
    labels = create_empty_label_data()
    return ImagesLabelsDataSet(images, labels)


def create_empty_image_data():
    buf = [0] * NUM_TRAINING_SAMPLES * IMAGE_WIDTH * IMAGE_HEIGHT
    data = numpy.fromiter(buf, dtype=numpy.uint8)
    return data.reshape(NUM_TRAINING_SAMPLES, IMAGE_WIDTH, IMAGE_HEIGHT, 1)


def create_empty_label_data():
    return create_empty_label_dataof_size(NUM_TRAINING_SAMPLES)


def create_empty_label_dataof_size(size):
    buf = [0] * size
    return numpy.fromiter(buf, dtype=numpy.uint8)
