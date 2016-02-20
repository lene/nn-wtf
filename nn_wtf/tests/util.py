import tensorflow as tf

__author__ = 'Lene Preuss <lp@sinnwerkstatt.com>'

MINIMAL_INPUT_SIZE = 2
MINIMAL_LAYER_GEOMETRY = (2, 2)
MINIMAL_OUTPUT_SIZE = 2
MINIMAL_BATCH_SIZE = 2


def create_minimal_input_placeholder():
    return tf.placeholder(tf.float32, shape=(MINIMAL_BATCH_SIZE, MINIMAL_INPUT_SIZE))


def get_project_root_folder():
    import os
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
