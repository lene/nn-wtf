from input_data import read_one_image_from_file

from mnist_graph import IMAGE_SIZE
import numpy

import unittest

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'


class InputDataTest(unittest.TestCase):

    def test_read_one_image_from_file(self):
        data = read_one_image_from_file('data/0.png')
        self.assertIsInstance(data, numpy.ndarray)
        self.assertEqual(4, len(data.shape))
        self.assertEqual(1, data.shape[0])
        self.assertEqual(IMAGE_SIZE, data.shape[1])
        self.assertEqual(IMAGE_SIZE, data.shape[2])
        self.assertEqual(1, data.shape[3])

        # print(data)