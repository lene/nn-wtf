from nn_wtf.neural_network_graph import NeuralNetworkGraph
from nn_wtf.saver_mixin import SaverMixin
from nn_wtf.tests.util import MINIMAL_LAYER_GEOMETRY, init_graph, train_neural_network, create_train_data_set

import unittest

import tensorflow as tf

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
        self.variables = {}

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

    def test_prediction_with_trained_graph(self):
        savefile = self._save_trained_graph()

        new_graph = init_graph(SavableNetwork(), session=tf.Session())
        for v in tf.all_variables():
            self.assertFalse(self.is_ndarray_equal(v.eval(new_graph.session), self.variables[v.op.name]))
        new_graph.restore()#savefile)
        for v in tf.all_variables():
            self.assertTrue(self.is_ndarray_equal(v.eval(new_graph.session), self.variables[v.op.name]))

    def _save_trained_graph(self):
        tf.reset_default_graph()
        with train_neural_network(create_train_data_set(), SavableNetwork()) as graph:
            for v in tf.all_variables():
                self.variables[v.op.name] = v.eval(graph.session)
            return graph.save(global_step=graph.trainer.num_steps())

    def _add_savefiles_to_list(self, savefile):
        self.generated_filenames.extend([savefile, '{}.meta'.format(savefile), 'checkpoint'])

    def is_ndarray_equal(self, array_1, array_2):
        import numpy
        return numpy.array_equal(array_1, array_2)
