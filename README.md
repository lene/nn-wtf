# nn-wtf - Neural Networks Wrapper for TensorFlow

nn-wtf aims at providing a convenience wrapper to Google's 
[TensorFlow](http://www.tensorflow.org/) machine learning library. 
Its focus is on making neural networks easy to set up, train and use.

The library is in pre-alpha right now and does not do anything useful yet.

## Installation

### Dependencies

You need to install TensorFlow manually. The installation process depends on 
your system. Install the version of TensorFlow built for Python 3.4. 

See https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html#download-and-setup
for details.

Example installation for Linux without GPU support:
```
$ pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.6.0-cp34-none-linux_x86_64.whl
```

### NN-WTF itself
Simple:
```
$ pip install nn_wtf
```

## Documentation

Sorry the documentation is absolutely minimal at this point. More useful
documentation will be ready by the time this package reaches alpha status.

### List of useful classes and methods

* `NeuralNetworkGraph`: Base class for defining and training neural networks
** `__init__(self, input_size, layer_sizes, output_size, learning_rate)`
** `set_session(self, session=None)`
** `train(self, data_sets, max_steps, precision, steps_between_checks, run_as_check, batch_size)`
** `get_predictor().predict(input_data)`
* `MNISTGraph`: Subclass of NeuralNetworkGraph suitable for working on MNIST data
* `NeuralNetworkOptimizer`: Optimize geometry of a neural network for fast training
** `__init__( self, tested_network, input_size, output_size, training_precision,
            layer_sizes, learning_rate, verbose, batch_size)`
** `brute_force_optimal_network_geometry(self, data_sets, max_steps)`

### Usage example

If you want to try it on MNIST data, try this sample program:

```python
from nn_wtf.input_data import read_data_sets, read_one_image_from_url
from nn_wtf.mnist_graph import MNISTGraph

import tensorflow as tf

data_sets = read_data_sets('.')
graph = MNISTGraph(
    learning_rate=0.1, layer_sizes=(64, 64, 16), train_dir='.'
)
graph.train(data_sets, max_steps=5000, precision=0.95)

image_data = read_one_image_from_url(
    'http://github.com/lene/nn-wtf/blob/master/nn_wtf/data/7_from_test_set.raw?raw=true'
)
prediction = graph.get_predictor().predict(image_data)
assert prediction == 7
```

From there on, you are on your own for now. More functionality and documentation
to come.