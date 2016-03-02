# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading MNIST data."""

import os

import numpy

import urllib.request


def maybe_download(filename, base_url, work_directory):
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    file_path = os.path.join(work_directory, filename)
    if not os.path.exists(file_path):
        file_path, _ = urllib.request.urlretrieve(base_url + filename, file_path)
        download_size = os.stat(file_path).st_size
        print('Successfully downloaded', filename, download_size, 'bytes.')
    return file_path


def read_images_from_file(filename, rows, cols, num_images, depth=1):
    _check_describes_image_geometry(rows, cols, depth)
    with open(filename, 'rb') as bytestream:
        return images_from_bytestream(bytestream, rows, cols, num_images, depth)


def read_images_from_files(rows, cols, depth, *filenames):
    _check_describes_image_geometry(rows, cols, depth)
    return concatenate_images_from_input_function(read_one_image_from_file, rows, cols, depth, filenames)


def read_images_from_urls(rows, cols, depth, *urls):
    _check_describes_image_geometry(rows, cols, depth)
    return concatenate_images_from_input_function(read_one_image_from_url, rows, cols, depth, urls)


def concatenate_images_from_input_function(input_function, rows, cols, depth, input_resources):
    image_data = numpy.concatenate(
        [input_function(input_resource, rows, cols, depth) for input_resource in input_resources]
    )
    return image_data


def read_images_from_url(url, rows, cols, num_images, depth=1):
    _check_describes_image_geometry(rows, cols, depth)
    with urllib.request.urlopen(url) as bytestream:
        return images_from_bytestream(bytestream, rows, cols, num_images, depth)


def images_from_bytestream(bytestream, rows, cols, num_images, depth=1):
    _check_describes_image_geometry(rows, cols, depth)
    buf = bytestream.read(rows * cols * depth * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    return data.reshape(num_images, rows, cols, depth)


def read_one_image_from_file(filename, rows, cols, depth=1):
    _check_describes_image_geometry(rows, cols, depth)
    with open(filename, 'rb') as bytestream:
        return one_image_from_bytestream(bytestream, rows, cols, depth)


def read_one_image_from_url(url, rows, cols, depth=1):
    _check_describes_image_geometry(rows, cols, depth)
    with urllib.request.urlopen(url) as bytestream:
        return one_image_from_bytestream(bytestream, rows, cols, depth)


def one_image_from_bytestream(bytestream, rows, cols, depth=1):
    _check_describes_image_geometry(rows, cols, depth)
    return images_from_bytestream(bytestream, rows, cols, depth)


def _check_describes_image_geometry(rows, cols, depth):
    assert rows > 0
    assert cols > 0
    assert 0 < depth < 3


