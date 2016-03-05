from nn_wtf.tests.util import MINIMAL_LAYER_GEOMETRY, create_train_data_sets
from nn_wtf.neural_network_graph import NeuralNetworkGraph

import tensorflow as tf

import unittest

from nn_wtf.trainer import Trainer

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'

STEPS_TO_RUN_TRAINER = 2
DEFAULT_LEARNING_RATE = 0.1


class TrainerTest(unittest.TestCase):

    def setUp(self):
        self.graph = NeuralNetworkGraph(2, MINIMAL_LAYER_GEOMETRY, 2)

    def test_create_trainer(self):
        trainer = Trainer(self.graph, DEFAULT_LEARNING_RATE)

    def test_create_trainer_with_invalid_graph(self):
        with self.assertRaises(AssertionError):
            Trainer(1)

    def test_train_runs(self):
        self.graph.init_trainer(DEFAULT_LEARNING_RATE)
        self.graph.set_session()
        self.graph.train(create_train_data_sets(), STEPS_TO_RUN_TRAINER)

    def test_train_fails_without_setting_session(self):
        self.graph.init_trainer(DEFAULT_LEARNING_RATE)
        with self.assertRaises(AssertionError):
            self.graph.train(create_train_data_sets(), STEPS_TO_RUN_TRAINER)

    def test_train_using_adagrad_optimizer_runs(self):
        self.graph.init_trainer(DEFAULT_LEARNING_RATE, tf.train.AdagradOptimizer)
        self.graph.set_session()
        self.graph.train(create_train_data_sets(), STEPS_TO_RUN_TRAINER)

    def test_train_using_adam_optimizer_runs(self):
        self.graph.init_trainer(DEFAULT_LEARNING_RATE, tf.train.AdamOptimizer)
        self.graph.set_session()
        self.graph.train(create_train_data_sets(), STEPS_TO_RUN_TRAINER)

    def test_train_using_ftrl_optimizer_runs(self):
        self.graph.init_trainer(DEFAULT_LEARNING_RATE, tf.train.FtrlOptimizer)
        self.graph.set_session()
        self.graph.train(create_train_data_sets(), STEPS_TO_RUN_TRAINER)

    def test_train_using_rmsprop_optimizer_runs(self):
        self.graph.init_trainer(DEFAULT_LEARNING_RATE, tf.train.RMSPropOptimizer)
        self.graph.set_session()
        self.graph.train(create_train_data_sets(), STEPS_TO_RUN_TRAINER)

    def test_train_using_momentum_optimizer_doesnt_run_without_momentum_parameter(self):
        with self.assertRaises(TypeError):
            self.graph.init_trainer(DEFAULT_LEARNING_RATE, tf.train.MomentumOptimizer)

    def test_train_using_momentum_optimizer_runs(self):
        self.graph.init_trainer(
            DEFAULT_LEARNING_RATE, tf.train.MomentumOptimizer, momentum=DEFAULT_LEARNING_RATE
        )
        self.graph.set_session()
        self.graph.train(create_train_data_sets(), STEPS_TO_RUN_TRAINER)

    def test_num_steps(self):
        self.test_train_runs()
        self.assertIsInstance(self.graph.trainer.num_steps(), int)
        self.assertLessEqual(STEPS_TO_RUN_TRAINER, self.graph.trainer.num_steps())
        self.assertGreater(self.graph.trainer.num_steps(), 0)
