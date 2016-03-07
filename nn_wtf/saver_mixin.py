from nn_wtf.neural_network_graph_mixin import NeuralNetworkGraphMixin, DEFAULT_TRAIN_DIR

import tensorflow as tf

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'


class SaverMixin(NeuralNetworkGraphMixin):

    def __init__(self, session, train_dir=DEFAULT_TRAIN_DIR):
        super().__init__(session, train_dir)
        self.saver = tf.train.Saver()

    def save(self, **kwargs):
        return self.saver.save(self.session, save_path=self.train_dir, **kwargs)

    def restore(self, save_path=None):
        if save_path is None:
            save_path = self.train_dir
        checkpoint = tf.train.get_checkpoint_state(save_path)
        if checkpoint is None:
            raise ValueError('No checkpoint found in {}'.format(save_path))
        self.saver.restore(self.session, checkpoint.model_checkpoint_path)


