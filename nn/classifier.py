import tensorflow as tf


class Classifier:
    def __init__(self):
        self.learning_rate = None
        self.epochs = None
        self.batch_size = None
        self.per_epoch = None
        self.convolution_layers = []
        self.layers = []
        self.save_path = None
        self.output_class_count = None
        self.convolution_size = None
        self.pooling_size = None
        self.optimiser = None
        self.accuracy = None
        self.init_operation = None
        self.saver = None
        self.weight_deviation = None
        self.bias_deviation = None

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

        return self

    def set_training_batches(self, epochs, batch_size, trainig_per_epoch):
        self.epochs = epochs
        self.batch_size = batch_size
        self.per_epoch = trainig_per_epoch

        return self

    def set_layers(self, convolution_size_list, neuron_list):
        self.convolution_layers = convolution_size_list
        self.layers = neuron_list

        return self

    def set_save_path(self, save_path):
        self.save_path = save_path

        return self

    def set_output_class_count(self, count):
        self.output_class_count = count

        return self

    def set_convolution_size(self, convolution_size, pooling_size):
        self.convolution_size = convolution_size
        self.pooling_size = pooling_size

        return self

    def set_initial_deviation(self, weight_deviation, bias_deviation):
        self.weight_deviation = weight_deviation
        self.bias_deviation = bias_deviation

        return self

    def initialize(self):
        pass

    def _create_convolution_layer(self, input_data, input_count, neuron_count, name):
        convolution_filter_shape = [
            self.convolution_size,
            self.convolution_size,
            input_count,
            neuron_count,
        ]

        weights = tf.Variable(
            tf.truncated_normal(convolution_filter_shape, stddev=self.weight_deviation),
            name=name + '_weights'
        )
        bias = tf.Variable(tf.truncated_normal([neuron_count]), name=name + '_bias')

        output_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME') + bias
        output_layer = tf.nn.relu(output_layer)

        # now perform max pooling
        ksize = [1, self.pooling_size, self.pooling_size, 1]
        strides = [1, self.pooling_size, self.pooling_size, 1] # No overlapping while pooling

        return tf.nn.max_pool(output_layer, ksize=ksize, strides=strides, padding='SAME')

    def _create_layer(self, input_data, input_count, neuron_count, name):
        weights = tf.Variable(
            tf.truncated_normal([input_count, neuron_count], stddev=self.weight_deviation),
            name=name + 'weights'
        )
        bias = tf.Variable(tf.truncated_normal([neuron_count], stddev=self.bias_deviation), name=name + 'bias')
        dense_layer = tf.matmul(input_data, weights) + bias

        return tf.nn.relu(dense_layer)

    def _get_output_prediction(self, value):
        if self.output_class_count > 1:
            argmax_value = tf.argmax(value, 0).eval()

            return (
                argmax_value,
                round(value[argmax_value] * 100, 2)
            )
        else:
            sigmoid_value = tf.sigmoid(value).eval()
            return (
                1 if sigmoid_value >= 0 else 0,
                round(abs(sigmoid_value * 100), 2)
            )
