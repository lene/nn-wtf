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
        self.layer_sizes = (self.input_size,) + layer_sizes
        self.num_hidden_layers = len(layer_sizes)
        self.output_size = output_size
        self.layers = []

    def build_neural_network(self, images):
        """Builds a neural network with the given layers and output size.

        Args:
          images: Images placeholder, from inputs().

        Returns:
          logits: Output tensor with the computed logits.
        """

        assert self.input_size == int(images.get_shape()[1])

        self.layers.append(images)
        for i in range(1, self.num_hidden_layers+1):
            self.layers.append(
                self.add_layer(
                    'layer%04d' % i, self.layer_sizes[i-1], self.layer_sizes[i], self.layers[i-1], tf.nn.relu
                )
            )

        logits = self.add_layer('output', self.layer_sizes[-1], self.output_size, self.layers[-1])

        return logits

    def loss(self, logits, labels):
        """Calculates the loss from the logits and the labels.

        Args:
          logits: Logits tensor, float - [batch_size, NUM_CLASSES].
          labels: Labels tensor, int32 - [batch_size].

        Returns:
          loss: Loss tensor of type float.
        """
        onehot_labels = self.convert_labels_to_onehot(labels)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits, onehot_labels, name='xentropy'
        )
        return tf.reduce_mean(cross_entropy, name='xentropy_mean')

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
        # For a classifier model, we can use the in_top_k Op.
        # It returns a bool tensor with shape [batch_size] that is true for
        # the examples where the label's is was in the top k (here k=1)
        # of all logits for that example.
        correct = tf.nn.in_top_k(logits, labels, 1)
        # Return the number of true entries.
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    def convert_labels_to_onehot(self, labels):
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

    def add_layer(self, layer_name, in_units_size, out_units_size, input_layer, function=lambda x: x):
        with tf.name_scope(layer_name):
            weights = self.initialize_weights(in_units_size, out_units_size)
            biases = self.initialize_biases(out_units_size)
            new_layer = function(tf.matmul(input_layer, weights) + biases)
        return new_layer

    def initialize_weights(self, in_units_size, out_units_size):
        return tf.Variable(
            tf.truncated_normal([in_units_size, out_units_size], stddev=1.0 / math.sqrt(float(in_units_size))),
            name='weights'
        )

    def initialize_biases(self, out_units_size):
        return tf.Variable(tf.zeros([out_units_size]), name='biases')

