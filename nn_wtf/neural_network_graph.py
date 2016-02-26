from nn_wtf.predictor import Predictor

import tensorflow as tf

import math

__author__ = 'Lene Preuss <lp@sinnwerkstatt.com>'


class NeuralNetworkGraph:

    def __init__(self, input_size, layer_sizes, output_size):
        """
        Args:
          input_size: number of input channels
          layer_sizes: Sizes of hidden layers in a tuple or list.
          output_size: Number of output channels.
        """
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.layer_sizes = self._set_layer_sizes(layer_sizes)
        self.num_hidden_layers = len(self.layer_sizes)-1
        self.layers = []
        self.predictor = None
        self.input_placeholder = None
        self.labels_placeholder = None

    def _set_layer_sizes(self, layer_sizes):
        layer_sizes = tuple(filter(None, layer_sizes))
        if layer_sizes[-1] < self.output_size:
            raise ValueError('Last layer size must be greater or equal output size')
        return (self.input_size,) + layer_sizes

    def build_neural_network(self, input_placeholder):
        """Builds a neural network with the given layers and output size.

        Args:
          input_placeholder: Images placeholder, from inputs().

        Returns:
          logits: Output tensor with the computed logits.
        """

        assert isinstance(input_placeholder, tf.Tensor), 'input placeholder not a tf.Tensor'
        assert self.input_size == int(input_placeholder.get_shape()[1]), \
            'declared input size {} not matching input placehoder shape {}'.format(
                self.input_size, int(input_placeholder.get_shape()[1])
            )
        assert self.layers == [], 'build_neural_network() has been called before'

        self.input_placeholder = input_placeholder

        self.layers.append(input_placeholder)
        for i in range(1, self.num_hidden_layers+1):
            self.layers.append(
                self._add_layer(
                    'layer%04d' % i, self.layer_sizes[i-1], self.layer_sizes[i], self.layers[i-1], tf.nn.relu
                )
            )

        logits = self._add_layer('output', self.layer_sizes[-1], self.output_size, self.layers[-1])

        self.layers.append(logits)

        return logits

    def build_train_ops(self, labels_placeholder, learning_rate):

        assert isinstance(labels_placeholder, tf.Tensor), 'labels placeholder not a tf.Tensor'
        assert len(self.layers) > 0, 'build_neural_network() needs to be called first'

        self.labels_placeholder = labels_placeholder

        # Add to the Graph the Ops for loss calculation.
        self.loss_op = self.loss(self.layers[-1], labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        self.train_op = self.training(self.loss_op, learning_rate)

        # Add the Op to compare the logits to the labels during evaluation.
        self.eval_correct_op = self.evaluation(self.layers[-1], labels_placeholder)

    def set_session(self, session=None):
        if session is None:
            session = tf.Session()
            init = tf.initialize_all_variables()
            session.run(init)
        self.session = session

    def train(
            self, data_sets, max_steps, precision=None, steps_between_checks=100, run_as_check=None,
            batch_size=1000
    ):
        self.step = 0
        while self.step < max_steps and not self._has_reached_precision(data_sets, precision, batch_size):

            feed_dict, loss_value = self.run_training_steps(data_sets, steps_between_checks, batch_size)
            self.step += steps_between_checks

            run_as_check(feed_dict, loss_value, self.step)

    def _has_reached_precision(self, data_sets, precision, batch_size):
        if precision is not None:
            self.do_eval(data_sets.test, batch_size)
            if self.precision > precision:
                return True
        return False

    def do_eval(self, data_set, batch_size):
        """Runs one evaluation against the full epoch of data.

        Args:
          session: The session in which the model has been trained.
          data_set: The set of images and labels to evaluate, from
            input_data.read_data_sets().
        """
        self.true_count = 0  # Counts the number of correct predictions.
        steps_per_epoch = data_set.num_examples // batch_size
        self.num_examples = steps_per_epoch * batch_size
        for _ in range(steps_per_epoch):
            feed_dict = self.fill_feed_dict(data_set, batch_size)
            self.true_count += self.session.run(self.eval_correct_op, feed_dict=feed_dict)
        self.precision = self.true_count / self.num_examples

    def run_training_steps(self, data_sets, num_steps, batch_size):
        feed_dict, loss_value = None, None
        for step in range(num_steps):
            feed_dict, loss_value = self.run_training_step(data_sets, batch_size)
        return feed_dict, loss_value

    def run_training_step(self, data_sets, batch_size):
        # Fill a feed dictionary with the actual set of images and labels for this particular
        # training step.
        feed_dict = self.fill_feed_dict(data_sets.train, batch_size)
        # Run one step of the model.  The return values are the activations from the `train_op`
        # (which is discarded) and the `loss` Op. To inspect the values of your Ops or
        # variables, you may include them in the list passed to session.run() and the value
        # tensors will be returned in the tuple from the call.
        _, loss_value = self.session.run([self.train_op, self.loss_op], feed_dict=feed_dict)
        return feed_dict, loss_value

    def save(self):
        pass

    def fill_feed_dict(self, data_set, batch_size):
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
        images_feed, labels_feed = data_set.next_batch(batch_size)
        feed_dict = {
            self.input_placeholder: images_feed,
            self.labels_placeholder: labels_feed,
        }
        return feed_dict

    def loss(self, logits, labels_placeholder):
        """Calculates the loss from the logits and the labels.

        Args:
          logits: Logits tensor, float - [batch_size, NUM_CLASSES].
          labels_placeholder: Labels tensor, int32 - [batch_size].

        Returns:
          loss: Loss tensor of type float.
        """
        onehot_labels = self._convert_labels_to_onehot(labels_placeholder)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits, onehot_labels, name='cross_entropy'
        )
        return tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

    def training(self, loss, learning_rate):
        """Sets up the training Ops.

        Creates a summarizer to track the loss over time in TensorBoard.

        Creates an optimizer and applies the gradients to all trainable variables.

        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train.

        Args:
          loss: Loss tensor, from loss().
          learning_rate: The learning rate to use for gradient descent.

        Returns:
          train_op: The Op for training.
      """
        # Add a scalar summary for the snapshot loss.
        tf.scalar_summary(loss.op.name, loss)
        # Create the gradient descent optimizer with the given learning rate.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # Create a variable to track the global step.
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # Use the optimizer to apply the gradients that minimize the loss
        # (and also increment the global step counter) as a single training step.
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    def evaluation(self, logits, labels):
        """Evaluate the quality of the logits at predicting the label.

        Args:
          logits: Logits tensor, float - [batch_size, NUM_CLASSES].
          labels: Labels tensor, int32 - [batch_size], with values in the
            range [0, NUM_CLASSES).

        Returns:
          A scalar int32 tensor with the number of examples (out of batch_size)
          that were predicted correctly.
        """
        # For a classifier model, we can use the in_top_k Op. It returns a bool
        # tensor with shape [batch_size] that is true for the examples where the
        # label's is was in the top k (here k=1) of all logits for that example.
        correct = tf.nn.in_top_k(logits, labels, 1)
        # Return the number of true entries.
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    def predict(self, session, image):
        if self.predictor is None:
            self.predictor = Predictor(self)
        return self.predictor.predict(session, image)

    def _add_layer(self, layer_name, in_units_size, out_units_size, input_layer, function=lambda x: x):
        with tf.name_scope(layer_name):
            weights = self._initialize_weights(in_units_size, out_units_size)
            biases = self._initialize_biases(out_units_size)
            new_layer = function(tf.matmul(input_layer, weights) + biases)
        return new_layer

    def _initialize_weights(self, in_units_size, out_units_size):
        """initialize weights with small amount of noise for symmetry breaking, and to prevent 0 gradients"""
        return tf.Variable(
            tf.truncated_normal([in_units_size, out_units_size], stddev=1.0 / math.sqrt(float(in_units_size))),
            name='weights'
        )

    def _initialize_biases(self, out_units_size):
        return tf.Variable(tf.zeros([out_units_size]), name='biases')

    def _convert_labels_to_onehot(self, labels):
        """
        Convert from sparse integer labels in the range [0, NUM_CLASSSES) to 1-hot dense float
        vectors (that is we will have batch_size vectors, each with NUM_CLASSES values, all of
        which are 0.0 except there will be a 1.0 in the entry corresponding to the label).
        """
        batch_size = tf.size(labels)
        labels = tf.expand_dims(labels, 1)
        indices = tf.expand_dims(tf.range(0, batch_size), 1)
        concated = tf.concat(1, [indices, labels])
        return tf.sparse_to_dense(concated, tf.pack([batch_size, self.output_size]), 1.0, 0.0)


