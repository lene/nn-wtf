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

    def prediction_probabilities(self, image):
        predictions = self._run_prediction_op(image, self.session)
        return predictions[1][0]

    def _run_prediction_op(self, image, session):
        image_data = image.reshape(self.graph.input_size)
        feed_dict = {self.graph.input_placeholder: [image_data]}
        best = session.run([self.prediction_op, self.probabilities_op], feed_dict)
        return best

    def _setup_prediction(self):
        if self.prediction_op is not None:
            return
        output = self.graph.output_layer()
        self.prediction_op = tf.argmax(output, 1)
        positive_output = output - tf.reduce_min(output, 1)
        self.probabilities_op = positive_output / tf.reduce_sum(positive_output, 1)
