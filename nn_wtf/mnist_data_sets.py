from nn_wtf.data_sets import DataSets
from nn_wtf.images_labels_data_set import ImagesLabelsDataSet
from nn_wtf.input_data import maybe_download, images_from_bytestream, \
    read_one_image_from_file, read_one_image_from_url, read_images_from_file, read_images_from_url, \
    read_images_from_files, read_images_from_urls

import numpy

import gzip

from nn_wtf.mnist_graph import MNISTGraph


__author__ = 'Lene Preuss <lene.preuss@gmail.com>'

"""DataSets for MNIST data."""


class MNISTDataSets(DataSets):

    SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
    VALIDATION_SIZE = 5000
    MNIST_MAGIC_LABELS = 2049
    MNIST_MAGIC_IMAGES = 2051

    def __init__(self, train_dir, one_hot=False):
        self.train_dir = train_dir
        self.one_hot = one_hot

        train_images = self.get_extracted_data(self.TRAIN_IMAGES, self.extract_images)
        train_labels = self.get_extracted_data(self.TRAIN_LABELS, self.extract_labels)
        test_images = self.get_extracted_data(self.TEST_IMAGES, self.extract_images)
        test_labels = self.get_extracted_data(self.TEST_LABELS, self.extract_labels)

        validation_images = train_images[:self.VALIDATION_SIZE]
        validation_labels = train_labels[:self.VALIDATION_SIZE]
        train_images = train_images[self.VALIDATION_SIZE:]
        train_labels = train_labels[self.VALIDATION_SIZE:]

        super().__init__(
            ImagesLabelsDataSet(train_images, train_labels),
            ImagesLabelsDataSet(validation_images, validation_labels),
            ImagesLabelsDataSet(test_images, test_labels)
        )

    @classmethod
    def read_one_image_from_file(cls, filename):
        return read_one_image_from_file(filename, MNISTGraph.IMAGE_SIZE, MNISTGraph.IMAGE_SIZE)

    @classmethod
    def read_one_image_from_url(cls, url):
        return read_one_image_from_url(url, MNISTGraph.IMAGE_SIZE, MNISTGraph.IMAGE_SIZE)

    @classmethod
    def read_images_from_file(cls, filename, num_images):
        return read_images_from_file(filename, MNISTGraph.IMAGE_SIZE, MNISTGraph.IMAGE_SIZE, num_images)

    @classmethod
    def read_images_from_url(cls, url, num_images):
        return read_images_from_url(url, MNISTGraph.IMAGE_SIZE, MNISTGraph.IMAGE_SIZE, num_images)

    @classmethod
    def read_images_from_files(cls, *filenames):
        return read_images_from_files(MNISTGraph.IMAGE_SIZE, MNISTGraph.IMAGE_SIZE, 1, *filenames)

    @classmethod
    def read_images_from_urls(cls, *filenames):
        return read_images_from_urls(MNISTGraph.IMAGE_SIZE, MNISTGraph.IMAGE_SIZE, 1, *filenames)

    def get_extracted_data(self, file_name, extract_function):
        local_file = maybe_download(file_name, self.SOURCE_URL, self.train_dir)
        return extract_function(local_file, one_hot=self.one_hot)

    def extract_images(self, filename, one_hot=False):
        """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
        print('Extracting', filename)
        with gzip.open(filename) as bytestream:
            magic = _read32(bytestream)
            if magic != self.MNIST_MAGIC_IMAGES:
                raise ValueError(
                    'Invalid magic number %d in MNIST image file: %s' %
                    (magic, filename))
            num_images = _read32(bytestream)
            rows = _read32(bytestream)
            cols = _read32(bytestream)
            return images_from_bytestream(bytestream, rows, cols, num_images)

    def extract_labels(self, filename, one_hot=False):
        """Extract the labels into a 1D uint8 numpy array [index]."""
        print('Extracting', filename)
        with gzip.open(filename) as bytestream:
            magic = _read32(bytestream)
            if magic != self.MNIST_MAGIC_LABELS:
                raise ValueError(
                    'Invalid magic number %d in MNIST label file: %s' %
                    (magic, filename))
            num_items = _read32(bytestream)
            buf = bytestream.read(num_items)
            labels = numpy.frombuffer(buf, dtype=numpy.uint8)
            if one_hot:
                return self.dense_to_one_hot(labels)
            return labels

    def dense_to_one_hot(self, labels_dense, num_classes=10):
        """Convert class labels from scalars to one-hot vectors."""
        num_labels = labels_dense.shape[0]
        index_offset = numpy.arange(num_labels) * num_classes
        labels_one_hot = numpy.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot



def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]

