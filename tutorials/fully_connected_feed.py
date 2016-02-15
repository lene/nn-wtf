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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time

import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tutorials import mnist


# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')


def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.

  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.

  Args:
    batch_size: The batch size will be baked into both placeholders.

  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         mnist.IMAGE_PIXELS))
  labels_placeholder = tf.placeholder(tf.int32, shape=batch_size)
  return images_placeholder, labels_placeholder




def run_training():
    """Train MNIST for a number of steps."""
    # Get the sets of images and labels for training, validation, and test on MNIST.
    data_sets = input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data)

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        graph = MNISTGraph()
        graph.run_training_graph(data_sets)

class MNISTGraph:

    def __init__(self):
        self._build_graph()
        self._setup_summaries()

    def _build_graph(self):
        # Generate placeholders for the images and labels.
        self.images_placeholder, self.labels_placeholder = placeholder_inputs(FLAGS.batch_size)
        # Build a Graph that computes predictions from the inference model.
        self.logits = mnist.inference(self.images_placeholder, FLAGS.hidden1, FLAGS.hidden2)
        # Add to the Graph the Ops for loss calculation.
        self.loss = mnist.loss(self.logits, self.labels_placeholder)
        # Add to the Graph the Ops that calculate and apply gradients.
        self.train_op = mnist.training(self.loss, FLAGS.learning_rate)
        # Add the Op to compare the logits to the labels during evaluation.
        self.eval_correct = mnist.evaluation(self.logits, self.labels_placeholder)

    def _setup_summaries(self):
        # Build the summary operation based on the TF collection of Summaries.
        self.summary_op = tf.merge_all_summaries()
        # Create a saver for writing training checkpoints.
        self.saver = tf.train.Saver()
        self.summary_writer = None

    def run_training_graph(self, data_sets):
        session = self.initialize_session()

        # And then after everything is built, start the training loop.
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()

            # Fill a feed dictionary with the actual set of images and labels for this particular
            # training step.
            feed_dict = self.fill_feed_dict(data_sets.train)

            # Run one step of the model.  The return values are the activations from the `train_op`
            # (which is discarded) and the `loss` Op. To inspect the values of your Ops or
            # variables, you may include them in the list passed to session.run() and the value
            # tensors will be returned in the tuple from the call.
            _, loss_value = session.run([self.train_op, self.loss], feed_dict=feed_dict)

            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often.
            if step % 100 == 0:
                self.write_summary(duration, feed_dict, loss_value, session, step)

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                self.saver.save(session, FLAGS.train_dir, global_step=step)
                # Evaluate against the training set.
                print('Training Data Eval:')
                self.do_eval(session, data_sets.train)
                # Evaluate against the validation set.
                print('Validation Data Eval:')
                self.do_eval(session, data_sets.validation)
                # Evaluate against the test set.
                print('Test Data Eval:')
                self.do_eval(session, data_sets.test)

    def initialize_session(self):
        # Create a session for running Ops on the Graph.
        sess = tf.Session()
        # Run the Op to initialize the variables.
        init = tf.initialize_all_variables()
        sess.run(init)
        # Instantiate a SummaryWriter to output summaries and the Graph.
        self.summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def=sess.graph_def)
        return sess

    def fill_feed_dict(self, data_set):
        """Fills the feed_dict for training the given step.

        A feed_dict takes the form of:
        feed_dict = {
            <placeholder>: <tensor of values to be passed for placeholder>,
              ....
        }

        Args:
          data_set: The set of images and labels, from input_data.read_data_sets()

        Returns:
          feed_dict: The feed dictionary mapping from placeholders to values.
        """
        # Create the feed_dict for the placeholders filled with the next `batch size ` examples.
        images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size, FLAGS.fake_data)
        feed_dict = {
            self.images_placeholder: images_feed,
            self.labels_placeholder: labels_feed,
        }
        return feed_dict

    def write_summary(self, duration, feed_dict, loss_value, sess, step):
        # Print status to stdout.
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        # Update the events file.
        summary_str = sess.run(self.summary_op, feed_dict=feed_dict)
        self.summary_writer.add_summary(summary_str, step)

    def do_eval(self, session, data_set):
        """Runs one evaluation against the full epoch of data.

        Args:
          session: The session in which the model has been trained.
          data_set: The set of images and labels to evaluate, from
            input_data.read_data_sets().
        """
        true_count = 0  # Counts the number of correct predictions.
        steps_per_epoch = data_set.num_examples // FLAGS.batch_size
        num_examples = steps_per_epoch * FLAGS.batch_size
        for _ in xrange(steps_per_epoch):
            feed_dict = self.fill_feed_dict(data_set)
            true_count += session.run(self.eval_correct, feed_dict=feed_dict)
        precision = true_count / num_examples
        print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
              (num_examples, true_count, precision))


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
