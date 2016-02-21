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

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
from random import randrange

from nn_wtf.images_labels_data_set import ImagesLabelsDataSet
from nn_wtf.neural_network_optimizer import NeuralNetworkOptimizer, timed_run
import nn_wtf.input_data as id
from nn_wtf.mnist_graph import MNISTGraph

import numpy
import tensorflow as tf

DEFAULT_OPTIMIZER_PRECISIONS = (0.9, 0.925, 0.95, 0.96, 0.97, 0.98, 0.99, 0.992)

# Basic model parameters as external flags.

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('self_test', False, 'Run self-test.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('training_precision', 0.0, 'Precision for geometry optimization runs.')
flags.DEFINE_float('desired_precision', 0.95, 'Desired training precision.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden3', None, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('batch_size', 100, 'Batch size. Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', '.nn_wtf-data', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data for unit testing.')
flags.DEFINE_boolean('list_precisions', False, 'If true, call optimizer for several precisions.')


def run_training():
    """Train MNIST for a number of steps."""
    # Get the sets of images and labels for training, validation, and test on MNIST.
    data_sets = id.fake_data_sets(False) if FLAGS.fake_data else id.read_data_sets(FLAGS.train_dir)

    geometry = get_network_geometry(data_sets)

    graph, cpu, wall = timed_run(run_final_training, geometry, data_sets)

    print(NeuralNetworkOptimizer.TimingInfo(cpu, wall, graph.precision, graph.step, geometry))

    return graph


def get_network_geometry(data_sets):
    if FLAGS.training_precision:
        optimizer = NeuralNetworkOptimizer(
            MNISTGraph, FLAGS.training_precision, learning_rate=FLAGS.learning_rate, verbose=True
        )
        geometry = optimizer.brute_force_optimal_network_geometry(data_sets, FLAGS.max_steps)
        print('Best geometry found:', geometry)
    else:
        geometry = (FLAGS.hidden1, FLAGS.hidden2, FLAGS.hidden3)

    return geometry


def run_final_training(geometry, data_sets):
    graph = MNISTGraph(
            learning_rate=FLAGS.learning_rate,
            hidden1=geometry[0], hidden2=geometry[1], hidden3=geometry[2],
            batch_size=FLAGS.batch_size, train_dir=FLAGS.train_dir
        )
    graph.train(data_sets, FLAGS.max_steps, precision=FLAGS.desired_precision, steps_between_checks=250)
    return graph


def main(_):
    if FLAGS.self_test:
        iterate_over_precisions(self_test=True)
    elif FLAGS.list_precisions:
        iterate_over_precisions(filename="results.txt")
    else:
        with tf.Graph().as_default():
            graph = run_training()
            image_data = id.read_one_image_from_file('nn_wtf/data/7_from_test_set.raw')
            prediction = graph.predict(image_data)
            print('actual number: 7, prediction:', prediction)
            # for i in range(10):
            #     image_data = id.read_one_image_from_file('nn_wtf/data/'+str(i)+'.raw')
            #     prediction = graph.predict(image_data)
            #     print(i, prediction)


def iterate_over_precisions(filename=None, self_test=False):
    precisions = (0.4,) if self_test else DEFAULT_OPTIMIZER_PRECISIONS
    layer_sizes = ((32,), (32,), (None,)) if self_test else None
    data_sets = id.read_data_sets(FLAGS.train_dir)
    final_results = {}
    for precision in precisions:
        with tf.Graph().as_default():
            optimizer = NeuralNetworkOptimizer(
                MNISTGraph, precision, layer_sizes=layer_sizes, learning_rate=0.1, verbose=True
            )
            results = optimizer.time_all_tested_geometries(data_sets, max(FLAGS.max_steps, 200000))
            final_results[precision] = results

    if filename:
        with open(filename, "w") as text_file:
            print(final_results, file=text_file)


if __name__ == '__main__':
    tf.app.run()
