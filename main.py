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

import tensorflow as tf

import input_data
from mnist_graph import MNISTGraph

# Basic model parameters as external flags.
from neural_network_optimizer import NeuralNetworkOptimizer, timed_run

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('training_precision', 0.9, 'Desired training precision.')
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

    optimizer = NeuralNetworkOptimizer(MNISTGraph, FLAGS, True)
    best_geometry = optimizer.brute_force_optimal_network_geometry(data_sets, FLAGS.training_precision)
    print(best_geometry)

    graph, timing_info = timed_run(run_final_training, best_geometry, data_sets)

    print(timing_info, graph.precision, graph.step)


def run_final_training(best_geometry, data_sets):
    with tf.Graph().as_default():
        graph = MNISTGraph(
            learning_rate=FLAGS.learning_rate,
            hidden1=best_geometry[3][0], hidden2=best_geometry[3][1], hidden3=best_geometry[3][2],
            batch_size=FLAGS.batch_size, train_dir=FLAGS.train_dir
        )
        graph.train(data_sets, FLAGS.max_steps, precision=FLAGS.desired_precision, steps_between_checks=250)
    return graph


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
