from nn_wtf.neural_network_graph import NeuralNetworkGraph
from nn_wtf.neural_network_graph_mixins import SaverMixin
from nn_wtf.tests.util import MINIMAL_LAYER_GEOMETRY, init_graph, train_neural_network, create_train_data_set

import unittest

from tempfile import gettempdir
from os import remove
from os.path import join

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'


class SavableNetwork(NeuralNetworkGraph, SaverMixin):
    def __init__(self):
        super().__init__(2, MINIMAL_LAYER_GEOMETRY, 2)

    def set_session(self, session=None, verbose=True, train_dir=gettempdir()):
        super().set_session()
        SaverMixin.__init__(self, self.session, train_dir)


class SaveAndRestoreTest(unittest.TestCase):

    def setUp(self):
        self.generated_filenames = []

    def tearDown(self):
        for filename in self.generated_filenames:
            remove(join(gettempdir(), filename))

    def test_save_untrained_network_runs(self):
        graph = init_graph(SavableNetwork())
        saved = graph.save(global_step=graph.trainer.num_steps())
        self._add_savefiles_to_list(saved)

    def test_save_trained_network_runs(self):
        graph = train_neural_network(create_train_data_set(), SavableNetwork())
        saved = graph.save(global_step=graph.trainer.num_steps())
        self._add_savefiles_to_list(saved)

    def _add_savefiles_to_list(self, savefile):
        self.generated_filenames.extend([savefile, '{}.meta'.format(savefile), 'checkpoint'])
