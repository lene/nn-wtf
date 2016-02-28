import tensorflow as tf

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'


def print_tensor(tensor):
    print(str(tensor.dtype) + ' Tensor ' + tensor.name + ' ' + str(tensor.get_shape()))


def print_tensors(tensors):
    for tensor in tensors:
        print_tensor(tensor)


def read_csv(file_name):
    reader = tf.TextLineReader(skip_header_lines=1)
    filename_queue = tf.train.string_input_producer([file_name])
    key, value = reader.read(filename_queue)
    record_defaults = [[1.], [1.], [1.]]

    col = tf.decode_csv(value, record_defaults=record_defaults)
    print_tensors(col)

    # features = tf.concat(0, [col1, col2])
    features = tf.pack([col[0], col[1]])
    print_tensor(features)

    with tf.Session() as sess:
        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(20):
            # Retrieve a single instance:
            example, label = sess.run([features, col[2]])
            print('example, label:', example, label)

    coord.request_stop()
    coord.join(threads)

    return key, value

c = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

print(read_csv('data/dummy_test_data.csv'))