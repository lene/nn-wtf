import numpy

__author__ = 'Lene Preuss <lp@sinnwerkstatt.com>'


class ImagesLabelsDataSet:

    def __init__(self, images, labels, fake_data=False, one_hot=False):
        """Construct a DataSet. one_hot arg is used only if fake_data is true.

        Args:
          images: 4D numpy.ndarray of shape (num images, image height, image width, image depth)
          labels: 1D numpy.ndarray of shape (num images)
        """

        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            _check_constructor_arguments_valid(images, labels)

            self._num_examples = images.shape[0]

            # Convert shape from [num examples, rows, columns, depth] to [num examples, rows*columns]
            # TODO: assumes depth == 1
            images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
            images = normalize(images)

        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            return self._fake_batch(batch_size)

        return self._next_batch_in_epoch(batch_size)

    def _next_batch_in_epoch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            self._shuffle_data()
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

    def _shuffle_data(self):
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self._images[perm]
        self._labels = self._labels[perm]

    def _fake_batch(self, batch_size):
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in range(batch_size)], [fake_label for _ in range(batch_size)]


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

