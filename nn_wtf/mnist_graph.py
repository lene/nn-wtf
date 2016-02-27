from nn_wtf.neural_network_graph import NeuralNetworkGraph
from nn_wtf.neural_network_graph_mixins import SaverMixin, DEFAULT_TRAIN_DIR, SummaryWriterMixin

__author__ = 'Lene Preuss <lp@sinnwerkstatt.com>'


# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


class MNISTGraph(NeuralNetworkGraph, SaverMixin, SummaryWriterMixin):

    def __init__(
        self, verbose=True,
        learning_rate=0.01, hidden1=128, hidden2=32, hidden3=None, batch_size=100,
        train_dir=DEFAULT_TRAIN_DIR
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        NeuralNetworkGraph.__init__(
            self, IMAGE_PIXELS, _get_geometry(hidden1, hidden2, hidden3), NUM_CLASSES
        )
        self.set_session()
        SaverMixin.__init__(self, self.session, train_dir)
        SummaryWriterMixin.__init__(self, self.session, verbose, train_dir)

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
        self.save(global_step=self.trainer.step)
        self.print_evaluations(data_sets, self.batch_size)


def _get_geometry(hidden1, hidden2, hidden3):
    hidden_layers = (hidden1, hidden2)
    if hidden3:
        hidden_layers += (hidden3,)
    return hidden_layers

