import tensorflow as tf

__author__ = 'Lene Preuss <lene.preuss@gmail.com>'


class Predictor:

    def __init__(self, graph, session):

        assert isinstance(session, tf.Session)

        self.graph = graph
        self.session = session
        self.prediction_op = None
        self.probabilities_op = None
        self._setup_prediction()

    def predict(self, image):
        predictions = self._run_prediction_op(image, self.session)
        return predictions[0][0]

    def predict_multiple(self, images, num):
        predictions = self._run_multi_prediction_op(images, num, self.session, [self.prediction_op])
        return predictions[0].tolist()

    def prediction_probabilities(self, image):
        predictions = self._run_prediction_op(image, self.session)
        return predictions[1][0]

    def _run_prediction_op(self, images, session):
        return self._run_multi_prediction_op(images, 1, session, [self.prediction_op, self.probabilities_op])

    def _run_multi_prediction_op(self, images, num_images, session, ops):
        image_data = images.reshape(num_images, self.graph.input_size)
        feed_dict = {self.graph.input_placeholder: image_data}
        best = session.run(ops, feed_dict)
        return best

    def _setup_prediction(self):
        if self.prediction_op is None:
            output = self.graph.output_layer()
            self.prediction_op = tf.argmax(output, 1)
            positive_output = output - tf.reduce_min(output, 1)
            self.probabilities_op = positive_output / tf.reduce_sum(positive_output, 1)
