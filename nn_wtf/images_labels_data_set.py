from nn_wtf.data_set_base import DataSetBase

import numpy

__author__ = 'Lene Preuss <lp@sinnwerkstatt.com>'


class ImagesLabelsDataSet(DataSetBase):

    def __init__(self, images, labels):
        """Construct a DataSet. one_hot arg is used only if fake_data is true.

        Args:
          images: 4D numpy.ndarray of shape (num images, image height, image width, image depth)
          labels: 1D numpy.ndarray of shape (num images)
        """

        _check_constructor_arguments_valid(images, labels)

        super().__init__(images, labels)

        self._num_examples = images.shape[0]

        # Convert shape from [num examples, rows, columns, depth] to [num examples, rows*columns]
        # TODO: assumes depth == 1
        images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
        images = normalize(images)

        self._input = images


def normalize(ndarray):
    """Transform a ndarray that contains uint8 values to floats between 0. and 1.

    :param ndarray:
    :return:
    """
    ndarray = ndarray.astype(numpy.float32)
    return numpy.multiply(ndarray, 1.0 / 255.0)

def invert(ndarray):
    return numpy.subtract(1.0, ndarray)

def _check_constructor_arguments_valid(images, labels):
    assert isinstance(images, numpy.ndarray), \
        'images not of type numpy.ndarray, but ' + type(images).__name__
    assert isinstance(labels, numpy.ndarray), \
        'labels not of type numpy.ndarray, but ' + type(images).__name__
    assert len(images.shape) == 4, \
        'images must have 4 dimensions: number of images, image height, image width, color depth'
    assert len(labels.shape) == 1, 'labels must have one dimension: number of labels'
    assert images.shape[0] == labels.shape[0], \
        'number of images: {}, number of labels: {}'.format(images.shape[0], labels.shape[0])
    assert images.shape[3] == 1, 'image depth must be 1'

