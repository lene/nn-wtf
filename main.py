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

from neural_network_optimizer import NeuralNetworkOptimizer, timed_run
import input_data
from mnist_graph import MNISTGraph

import tensorflow as tf

# Basic model parameters as external flags.

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('training_precision', 0.0, 'Precision for geometry optimization runs.')
flags.DEFINE_float('desired_precision', 0.95, 'Desired training precision.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Batch size. Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data for unit testing.')


def run_training():
    """Train MNIST for a number of steps."""
    # Get the sets of images and labels for training, validation, and test on MNIST.
    data_sets = input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data)

    geometry = get_network_geometry(data_sets)

    graph, cpu, wall = timed_run(run_final_training, geometry, data_sets)

    print(NeuralNetworkOptimizer.TimingInfo(cpu, wall, graph.precision, graph.step, geometry))

    return graph


def get_network_geometry(data_sets):
    if FLAGS.training_precision:
        optimizer = NeuralNetworkOptimizer(
            MNISTGraph, FLAGS.training_precision, FLAGS.learning_rate, verbose=True
        )
        geometry = optimizer.brute_force_optimal_network_geometry(data_sets, FLAGS.max_steps)
        print('Best geometry found:', geometry)
    else:
        geometry = (FLAGS.hidden1, FLAGS.hidden2)
    return geometry


def run_final_training(geometry, data_sets):
    with tf.Graph().as_default():
        graph = MNISTGraph(
            learning_rate=FLAGS.learning_rate,
            hidden1=geometry[0], hidden2=geometry[1], hidden3=geometry[2],
            batch_size=FLAGS.batch_size, train_dir=FLAGS.train_dir
        )
        graph.train(data_sets, FLAGS.max_steps, precision=FLAGS.desired_precision, steps_between_checks=250)
    return graph


def main(_):
    graph = run_training()


if __name__ == '__main__':
    tf.app.run()
