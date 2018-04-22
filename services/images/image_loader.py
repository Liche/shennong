import tensorflow as tf
from contextlib import contextmanager

class ImageLoader:
    def __init__(self, loader, image_dimensions):
        self.loader = loader
        self.image_dimensions = image_dimensions
        self.flattened_image_dimensions = self.image_dimensions[0] * self.image_dimensions[1] * self.image_dimensions[2]

    @contextmanager
    def load_image(self, filename):
        with self.loader.load(filename) as image_data:
            image_decoded = tf.image.decode_jpeg(image_data, channels=self.image_dimensions[2])
            image = tf.cast(image_decoded, tf.float32)

            resized_image = tf.image.resize_images([image], [self.image_dimensions[0], self.image_dimensions[1]])

            yield tf.reshape(resized_image, [self.flattened_image_dimensions])
