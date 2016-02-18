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
import time
import pprint

# Basic model parameters as external flags.
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
    best_geometry = brute_force_optimal_network_geometry(data_sets, FLAGS.training_precision)
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


def brute_force_optimal_network_geometry(data_sets, desired_precision, max_steps=10000):
    results = []
    for layer1_size in (32, 64, 96, 128):
        for layer2_size in (32, 64, 96, 128):
            if layer2_size > layer1_size:
                continue
            for layer3_size in (None, 32):
                run_info = timed_run_training(
                    data_sets,
                    layer1_size, layer2_size, layer3_size,
                    desired_precision=desired_precision, max_steps=max_steps
                )
                print(run_info)
                results.append(run_info)

    results = sorted(results, key=lambda r: r[0]['cpu_time'])
    pprint.pprint(results)
    return results[0]


def timed_run_training(data_sets, layer1_size, layer2_size, layer3_size, desired_precision=0.9, max_steps=10000):
    graph, timing = timed_run(run_training_once, data_sets, desired_precision, layer1_size, layer2_size, layer3_size, max_steps)
    return timing, graph.precision, graph.step, (layer1_size, layer2_size, layer3_size)

def run_training_once(data_sets, desired_precision, layer1_size, layer2_size, layer3_size, max_steps):
    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        graph = MNISTGraph(
            learning_rate=FLAGS.learning_rate, hidden1=layer1_size,
            hidden2=layer2_size, hidden3=layer3_size, batch_size=FLAGS.batch_size, train_dir=FLAGS.train_dir,
            verbose=False
        )
        graph.train(data_sets, max_steps, precision=desired_precision, steps_between_checks=50)
    return graph


def timed_run(function, *args, **kwargs):
    start_cpu_time, start_wall_time = time.process_time(), time.time()
    returned = function(*args, **kwargs)
    timing_info = {'cpu_time': time.process_time()-start_cpu_time, 'wall_time': time.time()-start_wall_time}
    return returned, timing_info

def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
