# nn-wtf - Neural Networks Wrapper for TensorFlow

nn-wtf aims at providing a convenience wrapper to Google's 
[TensorFlow](http://www.tensorflow.org/) machine learning library. 
Its focus is on making neural networks easy to set up, train and use.

The library is in pre-alpha right now and does not do anything useful yet.

If you want to try it on MNIST data though, try this sample program:

```python
from nn_wtf.input_data import read_data_sets, read_one_image_from_url
from nn_wtf.mnist_graph import MNISTGraph

import tensorflow as tf

data_sets = read_data_sets('.')
graph = MNISTGraph(
    learning_rate=0.1, hidden1=64, hidden2=64, hidden3=16, train_dir='.'
)
graph.train(data_sets, max_steps=5000, precision=0.95)

image_data = read_one_image_from_url(
    'http://github.com/lene/nn-wtf/blob/master/nn_wtf/data/7_from_test_set.raw?raw=true'
)
prediction = graph.predict(image_data)
assert prediction == 7
```

From there on, you are on your own for now. More functionality and documentation
to come.