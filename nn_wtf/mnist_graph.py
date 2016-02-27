import time
import math

import tensorflow as tf

from nn_wtf.neural_network_graph import NeuralNetworkGraph

__author__ = 'Lene Preuss <lp@sinnwerkstatt.com>'


# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

DEFAULT_TRAIN_DIR='.nn_wtf-data'


class MNISTGraph(NeuralNetworkGraph):

    def __init__(
        self, verbose=True,
        learning_rate=0.01, hidden1=128, hidden2=32, hidden3=None, batch_size=100,
        train_dir=DEFAULT_TRAIN_DIR
    ):
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.hidden = (hidden1, hidden2)
        if hidden3:
            self.hidden += (hidden3,)
        self.batch_size = batch_size
        self.train_dir = ensure_is_dir(train_dir)

        self.step = 0
        images_placeholder, labels_placeholder = placeholder_inputs(self.batch_size)

        super().__init__(images_placeholder.get_shape()[1], self.hidden, NUM_CLASSES)

        self.build_neural_network(images_placeholder)

        self.build_train_ops(labels_placeholder, self.learning_rate)

        self._setup_summaries()

        self.set_session()
        self.summary_writer = tf.train.SummaryWriter(self.train_dir, graph_def=self.session.graph_def)

    def train(
            self, data_sets, max_steps, precision=None, steps_between_checks=100, run_as_check=None,
            batch_size=1000
    ):
        # run write_summary() after every check, use self.batch_size as batch size
        super().train(
            data_sets, max_steps, precision, steps_between_checks,
            run_as_check=self.write_summary, batch_size=self.batch_size
        )

        # Save a checkpoint when done
        self.saver.save(self.session, save_path=self.train_dir, global_step=self.step)
        self.print_evaluations(data_sets)

    def print_evaluations(self, data_sets):
        if self.verbose: print('Training Data Eval:')
        self.print_eval(data_sets.train)

        if self.verbose: print('Validation Data Eval:')
        self.print_eval(data_sets.validation)

        if self.verbose: print('Test Data Eval:')
        self.print_eval(data_sets.test)

    def write_summary(self, feed_dict, loss_value, step):
        if self.verbose:
            print('Step %d: loss = %.2f ' % (step, loss_value))
        # Update the events file.
        summary_str = self.session.run(self.summary_op, feed_dict=feed_dict)
        self.summary_writer.add_summary(summary_str, step)

    def print_eval(self, data_set):
        if self.verbose:
            self.do_eval(data_set, self.batch_size)
            print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
                  (self.num_examples, self.true_count, self.precision))

    def evaluate_new_data_set(self, data_set):
        self.batch_size = data_set.num_examples
        self.num_examples = data_set.num_examples
        self.print_eval(data_set)

    def _setup_summaries(self):
        # Build the summary operation based on the TF collection of Summaries.
        self.summary_op = tf.merge_all_summaries()
        # Create a saver for writing training checkpoints.
        self.saver = tf.train.Saver()
        self.summary_writer = None


def ensure_is_dir(train_dir_string):
    if not train_dir_string[-1] == '/':
        train_dir_string += '/'
    return train_dir_string


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
    # Note that the shapes of the placeholders match the shapes of the full image and label
    # tensors, except the first dimension is now batch_size rather than the full size of
    # the train or test data sets.
    # images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_PIXELS))
    images_placeholder = tf.placeholder(tf.float32, shape=(None, IMAGE_PIXELS), name='images')
    # labels_placeholder = tf.placeholder(tf.int32, shape=batch_size)
    labels_placeholder = tf.placeholder(tf.int32, shape=(None,), name='labels')
    return images_placeholder, labels_placeholder

