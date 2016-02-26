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


class MNISTGraph:

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

        self._build_graph()
        self._setup_summaries()

    def train(self, data_sets, max_steps, precision=None, steps_between_checks=100):

        assert precision is None or 0. < precision < 1.

        self.session = self.initialize_session()
        self.graph.set_session(self.session)

        # And then after everything is built, start the training loop.
        self.graph.train(
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

    def initialize_session(self):
        # Create a session for running Ops on the Graph.
        sess = tf.Session()
        # Run the Op to initialize the variables.
        init = tf.initialize_all_variables()
        sess.run(init)
        # Instantiate a SummaryWriter to output summaries and the Graph.
        self.summary_writer = tf.train.SummaryWriter(self.train_dir, graph_def=sess.graph_def)
        return sess

    def write_summary(self, feed_dict, loss_value, step):
        # Print status to stdout.
        if self.verbose:
            print('Step %d: loss = %.2f ' % (step, loss_value))
        # Update the events file.
        summary_str = self.session.run(self.summary_op, feed_dict=feed_dict)
        self.summary_writer.add_summary(summary_str, step)


    def print_eval(self, data_set):
        self.graph.do_eval(data_set, self.batch_size)
        if self.verbose:
            print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
                  (self.graph.num_examples, self.graph.true_count, self.graph.precision))

    def evaluate_new_data_set(self, data_set):
        self.batch_size = data_set.num_examples
        self.num_examples = data_set.num_examples
        self.print_eval(data_set)

    def predict(self, image):
        return self.graph.predict(self.session, image)

    def prediction_probabilities(self, image):
        return self.graph.predictor.prediction_probabilities(self.session, image)

    def _build_graph(self):
        # Generate placeholders for the images and labels.
        self.images_placeholder, self.labels_placeholder = placeholder_inputs(self.batch_size)

        self.graph = NeuralNetworkGraph(
            self.images_placeholder.get_shape()[1], self.hidden, NUM_CLASSES
        )

        # Build a Graph that computes predictions from the inference model.
        self.graph.build_neural_network(self.images_placeholder)

        self.graph.build_train_ops(self.labels_placeholder, self.learning_rate)

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

