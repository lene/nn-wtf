from nn_wtf.neural_network_graph_mixin import NeuralNetworkGraphMixin, DEFAULT_TRAIN_DIR
from nn_wtf.trainer import Trainer

import tensorflow as tf

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'


class SummaryWriterMixin(NeuralNetworkGraphMixin):
    def __init__(self, session, verbose=False, train_dir=DEFAULT_TRAIN_DIR):
        super().__init__(session, train_dir)
        self.verbose = verbose
        self._setup_summaries()

    def write_summary(self, feed_dict, loss_value, step):
        if self.verbose:
            print('Step %d: loss = %.2f ' % (step, loss_value))
        # Update the events file.
        summary_str = self.session.run(self.summary_op, feed_dict=feed_dict)
        self.summary_writer.add_summary(summary_str, step)

    def print_evaluations(self, data_sets, batch_size):
        assert isinstance(self.trainer, Trainer), 'used SummaryMixin on a class other than NeuralNetworkGraph'
        self._print_eval(data_sets.train, batch_size, 'Training Data Eval:')
        self._print_eval(data_sets.validation, batch_size, 'Validation Data Eval:')
        self._print_eval(data_sets.test, batch_size, 'Test Data Eval:')

    def _print_eval(self, data_set, batch_size, message):
        if self.verbose:
            print(message)
            evaluation = self.trainer.do_eval(data_set, batch_size)
            print('  Num examples: {}  Num correct: {}  Precision @ 1: {:5.2f}'.format(
                evaluation.num_examples, evaluation.true_count, evaluation.precision)
            )

    def _setup_summaries(self):
        # Build the summary operation based on the TF collection of Summaries.
        self.summary_op = tf.merge_all_summaries()
        self.summary_writer = tf.train.SummaryWriter(self.train_dir, graph_def=self.session.graph_def)
