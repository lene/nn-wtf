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
        self.fake_data = False

        self._build_graph()
        self._setup_summaries()

    def train(self, data_sets, max_steps, precision=None, steps_between_checks=100):

        self.session = self.initialize_session()

        # And then after everything is built, start the training loop.
        for self.step in range(max_steps):
            start_time = time.time()

            # Fill a feed dictionary with the actual set of images and labels for this particular
            # training step.
            feed_dict = self.fill_feed_dict(data_sets.train)

            # Run one step of the model.  The return values are the activations from the `train_op`
            # (which is discarded) and the `loss` Op. To inspect the values of your Ops or
            # variables, you may include them in the list passed to session.run() and the value
            # tensors will be returned in the tuple from the call.
            _, loss_value = self.session.run([self.train_op, self.loss], feed_dict=feed_dict)

            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often.
            if self.step % steps_between_checks == 0:
                self.write_summary(duration, feed_dict, loss_value, self.step)
                if precision is not None:
                    self.do_eval(data_sets.test)
                    if self.precision > precision:
                        return

            # Save a checkpoint and evaluate the model periodically.
            if (self.step + 1) % 1000 == 0 or (self.step + 1) == max_steps:
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
        images_feed, labels_feed = data_set.next_batch(self.batch_size, self.fake_data)
        feed_dict = {
            self.images_placeholder: images_feed,
            self.labels_placeholder: labels_feed,
        }
        return feed_dict

    def write_summary(self, duration, feed_dict, loss_value, step):
        # Print status to stdout.
        if self.verbose:
            print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        # Update the events file.
        summary_str = self.session.run(self.summary_op, feed_dict=feed_dict)
        self.summary_writer.add_summary(summary_str, step)

    def do_eval(self, data_set):
        """Runs one evaluation against the full epoch of data.

        Args:
          session: The session in which the model has been trained.
          data_set: The set of images and labels to evaluate, from
            input_data.read_data_sets().
        """
        self.true_count = 0  # Counts the number of correct predictions.
        steps_per_epoch = data_set.num_examples // self.batch_size
        self.num_examples = steps_per_epoch * self.batch_size
        for _ in range(steps_per_epoch):
            feed_dict = self.fill_feed_dict(data_set)
            self.true_count += self.session.run(self.eval_correct, feed_dict=feed_dict)
        self.precision = self.true_count / self.num_examples

    def print_eval(self, data_set):
        self.do_eval(data_set)
        if self.verbose:
            print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
                  (self.num_examples, self.true_count, self.precision))

    def evaluate_new_data_set(self, data_set):
        self.batch_size = data_set.num_examples
        self.num_examples = data_set.num_examples
        self.print_eval(data_set)

    def predict(self, image):
        return self.graph.predict(self.session, image)

    def _build_graph(self):
        # Generate placeholders for the images and labels.
        self.images_placeholder, self.labels_placeholder = placeholder_inputs(self.batch_size)

        self.graph = NeuralNetworkGraph(
            self.images_placeholder.get_shape()[1], self.hidden, NUM_CLASSES
        )

        # Build a Graph that computes predictions from the inference model.
        self.logits = self.graph.build_neural_network(
            self.images_placeholder
        )

        # Add to the Graph the Ops for loss calculation.
        self.loss = self.graph.loss(self.logits, self.labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        self.train_op = self.graph.training(self.loss, self.learning_rate)

        # Add the Op to compare the logits to the labels during evaluation.
        self.eval_correct = self.graph.evaluation(self.logits, self.labels_placeholder)

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

