from nn_wtf.neural_network_graph import NeuralNetworkGraph
from nn_wtf.saver_mixin import SaverMixin, DEFAULT_TRAIN_DIR
from nn_wtf.summary_writer_mixin import SummaryWriterMixin

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'


class MNISTGraph(NeuralNetworkGraph, SaverMixin, SummaryWriterMixin):
    """A NeuralNetworkGraph configured with the MNIST data geometry which also
    periodically writes a report during the training process and saves the
    trained graph.
    """

    # The MNIST dataset has 10 classes, representing the digits 0 through 9.
    NUM_CLASSES = 10

    # The MNIST images are always 28x28 pixels.
    IMAGE_SIZE = 28
    IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

    def __init__(
        self, input_size=None, layer_sizes=(128, 32, None), output_size=None):
        """The MNISTGraph constructor takes no positional args, in contrast with NeuralNetworkGraph.
        input_size and output_size are present for compatibility, but ignored.

        :param input_size: ignored, present for client compatibility
        :param layer_sizes: tuple of sizes of the neural network hidden layers
        :param output_size: ignored, present for client compatibility
        :param learning_rate: learning rate for gradient descent optimizer
        :param verbose: whether to print some info about the training progress
        :param train_dir: where to write savepoints and summaries
        """
        NeuralNetworkGraph.__init__(
            self, self.IMAGE_PIXELS, layer_sizes, self.NUM_CLASSES
        )

    def set_session(self, session=None, verbose=True, train_dir=DEFAULT_TRAIN_DIR):
        super().set_session()
        SaverMixin.__init__(self, self.session, train_dir)
        SummaryWriterMixin.__init__(self, self.session, verbose, train_dir)

    def train(
            self, data_sets, max_steps, precision=None, steps_between_checks=100, run_as_check=None,
            batch_size=1000
    ):
        assert self.summary_op is not None, 'called train() before setting up summary op'
        assert self.summary_writer is not None, 'called train() before setting up summary writer'
        assert self.saver is not None, 'called train() before setting up saver'

        # run write_summary() after every check
        super().train(
            data_sets, max_steps, precision, steps_between_checks,
            run_as_check=self.write_summary, batch_size=batch_size
        )

        # Save a checkpoint when done
        self.save(global_step=self.trainer.num_steps())
        self.print_evaluations(data_sets, batch_size)

