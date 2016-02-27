import time
import math

import tensorflow as tf

from nn_wtf.neural_network_graph import NeuralNetworkGraph, CHANGE_THIS_LEARNING_RATE
from nn_wtf.trainer import Trainer

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

        super().__init__(IMAGE_PIXELS, self.hidden, NUM_CLASSES)

        self.set_session()
        self._setup_summaries()

    def train(
            self, data_sets, max_steps, precision=None, steps_between_checks=100, run_as_check=None,
            batch_size=1000
    ):
        assert self.summary_op is not None, 'called train() before setting up summary op'
        assert self.summary_writer is not None, 'called train() before setting up summary writer'
        assert self.saver is not None, 'called train() before setting up saver'

        # run write_summary() after every check, use self.batch_size as batch size
        super().train(
            data_sets, max_steps, precision, steps_between_checks,
            run_as_check=self.write_summary, batch_size=self.batch_size
        )

        # Save a checkpoint when done
        self.saver.save(self.session, save_path=self.train_dir, global_step=self.step)
        self.print_evaluations(data_sets)

    def print_evaluations(self, data_sets):
        self.print_eval(data_sets.train, 'Training Data Eval:')
        self.print_eval(data_sets.validation, 'Validation Data Eval:')
        self.print_eval(data_sets.test, 'Test Data Eval:')

    def write_summary(self, feed_dict, loss_value, step):
        assert self.summary_op is not None

        if self.verbose:
            print('Step %d: loss = %.2f ' % (step, loss_value))
        # Update the events file.
        summary_str = self.session.run(self.summary_op, feed_dict=feed_dict)
        self.summary_writer.add_summary(summary_str, step)

    def print_eval(self, data_set, message):
        if self.verbose:
            print(message)
            self.trainer.do_eval(data_set, self.batch_size)
            print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
                  (self.trainer.num_examples, self.trainer.true_count, self.trainer.precision))

    def _setup_summaries(self):
        # Build the summary operation based on the TF collection of Summaries.
        self.summary_op = tf.merge_all_summaries()
        # Create a saver for writing training checkpoints.
        self.saver = tf.train.Saver()
        self.summary_writer = tf.train.SummaryWriter(self.train_dir, graph_def=self.session.graph_def)


def ensure_is_dir(train_dir_string):
    if not train_dir_string[-1] == '/':
        train_dir_string += '/'
    return train_dir_string

