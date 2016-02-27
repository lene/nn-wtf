import tensorflow as tf

from nn_wtf.trainer import Trainer

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'

DEFAULT_TRAIN_DIR = '.nn_wtf-data'


class NeuralNetworkGraphMixin:
    def __init__(self, session, train_dir=DEFAULT_TRAIN_DIR):
        assert isinstance(session, tf.Session), 'session must be set when initializing saver'
        self.session = session
        self.train_dir = ensure_is_dir(train_dir)


def ensure_is_dir(train_dir_string):
    if not train_dir_string[-1] == '/':
        train_dir_string += '/'
    return train_dir_string


class SaverMixin(NeuralNetworkGraphMixin):

    def __init__(self, session, train_dir=DEFAULT_TRAIN_DIR):
        super().__init__(session, train_dir)
        # Create a saver for writing training checkpoints.
        self.saver = tf.train.Saver()

    def save(self, **kwargs):
        self.saver.save(self.session, save_path=self.train_dir, **kwargs)


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
            self.trainer.do_eval(data_set, batch_size)
            print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
                  (self.trainer.num_examples, self.trainer.true_count, self.trainer.precision))

    def _setup_summaries(self):
        # Build the summary operation based on the TF collection of Summaries.
        self.summary_op = tf.merge_all_summaries()
        self.summary_writer = tf.train.SummaryWriter(self.train_dir, graph_def=self.session.graph_def)
