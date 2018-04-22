import os
import random
import tensorflow as tf
from nn.classifier import Classifier
from services.images.image_loader import ImageLoader
from services.file.local_loader import LocalLoader
from services.file.url_loader import UrlLoader


class ImageClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.image_dimensions = (0, 0, 0)
        self.flattened_image_dimensions = 0

        self.training_set_path = ''
        self.testing_set_path = ''

        self.input = None
        self.shaped_input = None
        self.output = None
        self.output_prediction = None
        self.cross_entropy = None

        self.image_loader = None
        self.training_image_loader = None

        random.seed()

    def set_training_set_path(self, training_path):
        self.training_set_path = training_path

        return self

    def set_testing_set_path(self, testing_path):
        self.testing_set_path = testing_path

        return self

    def set_image_dimensions(self, width, height, colors):
        self.image_dimensions = (width, height, colors)
        self.flattened_image_dimensions = width * height * colors

        return self

    def load_training_batch(self, batch_size):
        image_list = []
        label_list = []

        for counter in range(batch_size):
            current_label = random.randint(0, self.output_class_count - 1)

            training_label_directory = f'{self.training_set_path}/{current_label}'
            file_in_path = os.listdir(training_label_directory)
            image_filename = random.choice(file_in_path)
            with self.training_image_loader.load_image(f'{training_label_directory}/{image_filename}') as image:
                image_list.append(image.eval())
            label = [0] * self.output_class_count
            label[current_label] = 1

            label_list.append(label)

        return image_list, label_list

    def load_testing_set(self):
        image_list = []
        label_list = []
        test_output_classes = os.listdir(self.testing_set_path)
        for image_class in test_output_classes:
            directory = f'{self.testing_set_path}/{image_class}'
            files = os.listdir(directory)
            for filename in files:
                filepath = f'{directory}/{filename}'
                with self.training_image_loader.load_image(filepath) as image:
                    image_list.append(image.eval())

                label = [0] * self.output_class_count
                label[int(image_class)] = 1

                label_list.append(label)

        return image_list, label_list

    def _create_input_layer(self):
        self.input = tf.placeholder(tf.float32, [None, self.flattened_image_dimensions])
        self.shaped_input = tf.reshape(self.input, [-1] + list(self.image_dimensions))

    def _create_output_layer(self, input_layer, input_count):
        self.output = tf.placeholder(tf.float32, [None, self.output_class_count])
        weights = tf.Variable(
            tf.truncated_normal([input_count, self.output_class_count], stddev=self.weight_deviation),
            name='output_weights'
        )
        bias = tf.Variable(
            tf.truncated_normal([self.output_class_count], stddev=self.bias_deviation),
            name='output_bias'
        )
        return tf.matmul(input_layer, weights) + bias

    def initialize(self):
        self.training_image_loader = ImageLoader(LocalLoader(), self.image_dimensions)
        self.image_loader = ImageLoader(UrlLoader(), self.image_dimensions)
        self._create_input_layer()

        input_layer = self.shaped_input
        input_count = self.image_dimensions[2]
        for index, neuron_count in enumerate(self.convolution_layers):
            input_layer = self._create_convolution_layer(input_layer, input_count, neuron_count, f'c_layer{index}')
            input_count = neuron_count

        flattened_width = int(self.image_dimensions[0] / (self.pooling_size * len(self.convolution_layers)))
        flattened_height = int(self.image_dimensions[1] / (self.pooling_size * len(self.convolution_layers)))

        input_count = flattened_width * flattened_height * input_count
        input_layer = tf.reshape(input_layer, [-1, input_count])

        for index, neuron_count in enumerate(self.layers):
            input_layer = self._create_layer(input_layer, input_count, neuron_count, f'layer{index}')
            input_count = neuron_count

        self._create_output_layer(input_layer, input_count)

        output_dense_layer = self._create_output_layer(input_layer, input_count)

        self.output_prediction = tf.nn.softmax(output_dense_layer)

        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_dense_layer, labels=self.output)
        )

        # add an optimiser
        self.optimiser = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cross_entropy)

        # define an accuracy assessment operation
        correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.output_prediction, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.init_operation = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def train(self):
        with tf.Session() as session:
            session.run(self.init_operation)
            for epoch in range(self.epochs):
                average_cost = 0
                for counter in range(self.per_epoch):
                    images, labels = self.load_training_batch(self.batch_size)
                    _, c = session.run(
                        [self.optimiser, self.cross_entropy],
                        feed_dict={self.input: images, self.output: labels}
                    )
                    average_cost += c / self.per_epoch
                print(f'Epoch: {epoch + 1}, Cost: {average_cost}')
                tf.logging.info(f'Epoch: {epoch + 1}, Cost: {average_cost}')

            print(f'Training Complete')
            tf.logging.info('Training Complete')
            self.saver.save(session, self.save_path)

            test_images, test_labels = self.load_testing_set()
            test_accuracy = session.run(self.accuracy, feed_dict={self.input: test_images, self.output: test_labels})
            print(f'Test Accuracy: {test_accuracy}')
            tf.logging.info(f'Test Accuracy: {test_accuracy}')

    def classify(self, images):
        with tf.Session() as session:
            self.saver.restore(session, self.save_path)

            image_list = []
            for image_string in images:
                with self.image_loader.load_image(image_string) as image:
                    image_list.append(image.eval())

            output_prediction = session.run(self.output_prediction, feed_dict={self.input: image_list})

            output = []

            for prediction in output_prediction:
                output.append(self._get_output_prediction(prediction))

            return output
