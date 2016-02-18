import time
import pprint
import tensorflow as tf

__author__ = 'Lene Preuss <lp@sinnwerkstatt.com>'


class NeuralNetworkOptimizer:

    def __init__(self, tested_network, FLAGS, verbose=False):
        self.tested_network = tested_network
        self.verbose = verbose
        self.FLAGS = FLAGS

    def brute_force_optimal_network_geometry(self, data_sets, desired_precision, max_steps=10000):
        results = []
        for layer1_size in (32, 64, 96, 128):
            for layer2_size in (32, 64, 96, 128):
                if layer2_size > layer1_size:
                    continue
                for layer3_size in (None, 32):
                    run_info = self.timed_run_training(
                        data_sets,
                        layer1_size, layer2_size, layer3_size,
                        desired_precision=desired_precision, max_steps=max_steps
                    )
                    if self.verbose: print(run_info)
                    results.append(run_info)

        results = sorted(results, key=lambda r: r[0]['cpu_time'])
        if self.verbose: pprint.pprint(results)
        return results[0]

    def timed_run_training(self, data_sets, layer1_size, layer2_size, layer3_size, desired_precision=0.9, max_steps=10000):
        graph, timing = timed_run(self.run_training_once, data_sets, desired_precision, layer1_size, layer2_size, layer3_size, max_steps)
        return timing, graph.precision, graph.step, (layer1_size, layer2_size, layer3_size)

    def run_training_once(self, data_sets, desired_precision, layer1_size, layer2_size, layer3_size, max_steps):
        # Tell TensorFlow that the model will be built into the default Graph.
        with tf.Graph().as_default():
            graph = self.tested_network(
                learning_rate=self.FLAGS.learning_rate,
                hidden1=layer1_size,
                hidden2=layer2_size, hidden3=layer3_size,
                batch_size=self.FLAGS.batch_size, train_dir=self.FLAGS.train_dir,
                verbose=False
            )
            graph.train(data_sets, max_steps, precision=desired_precision, steps_between_checks=50)
        return graph


def timed_run(function, *args, **kwargs):
        start_cpu_time, start_wall_time = time.process_time(), time.time()
        returned = function(*args, **kwargs)
        timing_info = {'cpu_time': time.process_time()-start_cpu_time, 'wall_time': time.time()-start_wall_time}
        return returned, timing_info
