# nn-wtf - Neural Networks Wrapper for TensorFlow

nn-wtf aims at providing a convenience wrapper to Google's 
[TensorFlow](http://www.tensorflow.org/) machine learning library. 
Its focus is on making neural networks easy to set up, train and use.

The library is in pre-alpha right now and does not do anything seriously useful 
yet.

## Installation

nn-wtf runs under Python3.4 and above.

### Dependencies

Depending on whether you want GPU support or not, you need to do either 
```
$ pip install tensorflow
```
or
```
$ pip install tensorflow-gpu
```

See https://www.tensorflow.org/get_started/os_setup for details.

The GPU version of TensorFlow needs the Nvidia CUDA libraries and the cuDNN Framework installed. See
[TensorFlow-GPU dependencies](#tensorflow-gpu-dependencies) for instructions.

### NN-WTF itself
Simple:
```
$ pip install nn_wtf
```

### TensorFlow-GPU dependencies
The GPU version of TensorFlow needs the CUDA Toolkit 8.0 and cuDNN 5. There are instructions on how 
 to do this at [Install CUDA (GPUs on Linux)](https://www.tensorflow.org/get_started/os_setup#optional_install_cuda_gpus_on_linux)

A brief summary for Ubuntu Linux:
* Download the CUDA 8 .deb from https://developer.nvidia.com/cuda-downloads and follow the 
  installation instructions.
* Download the cuDNN 5 archive from https://developer.nvidia.com/rdp/cudnn-download.
  * You will have to register and set up an NVIDIA Developer account and apply for membership in the
    Accelerated Computing Developer Program (https://developer.nvidia.com/rdp/cuda-registered-developer-program)
    to download cuDNN.
* Install cuDNN by extracting the downloaded archive and copying the extracted files to 
  `/usr/local/cuda/include` and `/usr/local/cuda/lib64`.
* Verify the installation by opening a Python interpreter and typing
```python
>>> import tensorflow as tf
```
If no errors occur, your installation is successful.

## Documentation

Sorry the documentation is absolutely minimal at this point. More useful
documentation will be ready by the time this package reaches alpha status.

### List of useful classes and methods

* `NeuralNetworkGraph`: Base class for defining and training neural networks
  * `__init__(self, input_size, layer_sizes, output_size, learning_rate)`
  * `set_session(self, session=None)`
  * `train(self, data_sets, max_steps, precision, steps_between_checks, run_as_check, batch_size)`
  * `get_predictor().predict(input_data)`
* `MNISTGraph`: Subclass of NeuralNetworkGraph suitable for working on MNIST data
* `NeuralNetworkOptimizer`: Optimize geometry of a neural network for fast training
  * `__init__( self, tested_network, input_size, output_size, training_precision,
            layer_sizes, learning_rate, verbose, batch_size)`
  * `brute_force_optimal_network_geometry(self, data_sets, max_steps)`

### Usage example

If you want to try it on MNIST data, try this sample program:

```python
from nn_wtf.mnist_data_sets import MNISTDataSets
from nn_wtf.mnist_graph import MNISTGraph

import tensorflow as tf

data_sets = MNISTDataSets('.')
graph = MNISTGraph(
    learning_rate=0.1, layer_sizes=(64, 64, 16), train_dir='.'
)
graph.train(data_sets, max_steps=5000, precision=0.95)

# verify the training worked by testing if a handwritten number is recognized
image_data = MNISTDataSets.read_one_image_from_url(
    'http://github.com/lene/nn-wtf/blob/master/nn_wtf/data/7_from_test_set.raw?raw=true'
)
prediction = graph.get_predictor().predict(image_data)
assert prediction == 7
```

If you have cloned or downloaded the git repository at https://github.com/lene/nn-wtf, you can 
also have a look at `main.py` in the root directory for some example usage.

From there on, you are on your own for now. More functionality and documentation
to come.

