import time
import math

import tensorflow as tf

import mnist
from mnist import NUM_CLASSES

__author__ = 'Lene Preuss <lp@sinnwerkstatt.com>'


class MNISTGraph:

    def __init__(
            self,
            learning_rate=0.01, max_steps=2000, hidden1=128, hidden2=32,
            batch_size=100, train_dir='data'
    ):
        # self.flags = flags
        self.learning_rate = learning_rate
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.batch_size = batch_size
        self.train_dir = train_dir
        self.fake_data = False

        self._build_graph()
        self._setup_summaries()

    def train(self, data_sets, max_steps):

        session = self.initialize_session()

        # And then after everything is built, start the training loop.
        for step in range(max_steps):
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
            if (step + 1) % 1000 == 0 or (step + 1) == max_steps:
                self.saver.save(session, self.train_dir, global_step=step)
                self.print_evaluations(data_sets, session)

    def print_evaluations(self, data_sets, session):
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
        steps_per_epoch = data_set.num_examples // self.batch_size
        num_examples = steps_per_epoch * self.batch_size
        for _ in range(steps_per_epoch):
            feed_dict = self.fill_feed_dict(data_set)
            true_count += session.run(self.eval_correct, feed_dict=feed_dict)
        precision = true_count / num_examples
        print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
              (num_examples, true_count, precision))

    def _build_graph(self):
        # Generate placeholders for the images and labels.
        self.images_placeholder, self.labels_placeholder = placeholder_inputs(self.batch_size)

        # Build a Graph that computes predictions from the inference model.
        self.logits = build_neural_network(
            self.images_placeholder,
            (self.hidden1, self.hidden2),
            NUM_CLASSES
        )

        # Add to the Graph the Ops for loss calculation.
        self.loss = mnist.loss(self.logits, self.labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        self.train_op = mnist.training(self.loss, self.learning_rate)

        # Add the Op to compare the logits to the labels during evaluation.
        self.eval_correct = mnist.evaluation(self.logits, self.labels_placeholder)

    def _setup_summaries(self):
        # Build the summary operation based on the TF collection of Summaries.
        self.summary_op = tf.merge_all_summaries()
        # Create a saver for writing training checkpoints.
        self.saver = tf.train.Saver()
        self.summary_writer = None


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
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         mnist.IMAGE_PIXELS))
  labels_placeholder = tf.placeholder(tf.int32, shape=batch_size)
  return images_placeholder, labels_placeholder


def build_neural_network(images, hidden_layer_sizes, output_size):
    """Build the MNIST model up to where it may be used for inference.

    Args:
      images: Images placeholder, from inputs().
      hidden1_units: Size of the first hidden layer.
      hidden2_units: Size of the second hidden layer.

    Returns:
      softmax_linear: Output tensor with the computed logits.
    """
    input_size = int(images.get_shape()[1])
    # hidden1 = add_layer('hidden1', IMAGE_PIXELS, hidden_layer_sizes[0], images, tf.nn.relu)
    hidden1 = add_layer('hidden1', input_size, hidden_layer_sizes[0], images, tf.nn.relu)
    hidden2 = add_layer('hidden2', hidden_layer_sizes[0], hidden_layer_sizes[1], hidden1, tf.nn.relu)
    logits = add_layer('softmax_linear', hidden_layer_sizes[1], output_size, hidden2)
    return logits


def add_layer(layer_name, in_units_size, out_units_size, input_layer, function=lambda x: x):
    with tf.name_scope(layer_name):
        weights = initialize_weights(in_units_size, out_units_size)
        biases = initialize_biases(out_units_size)
        new_layer = function(tf.matmul(input_layer, weights) + biases)
    return new_layer


def initialize_weights(in_units_size, out_units_size):
    return tf.Variable(
        tf.truncated_normal([in_units_size, out_units_size], stddev=1.0 / math.sqrt(float(in_units_size))),
        name='weights'
    )


def initialize_biases(out_units_size):
    return tf.Variable(tf.zeros([out_units_size]), name='biases')


