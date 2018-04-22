import os
from nn.image_classifier import ImageClassifier
from services.config.loader import Loader as ConfigLoader


dir_path = os.path.dirname(os.path.realpath(__file__))
config_loader = ConfigLoader(f'{dir_path}/config.yml')

classifier = ImageClassifier()
classifier.set_learning_rate(0.0001).set_training_batches(10, 50, 5)
classifier.set_convolution_size(5, 2)
classifier.set_image_dimensions(100, 100, 3)
classifier.set_output_class_count(2)
classifier.set_save_path(config_loader.get('save.path'))
classifier.set_training_set_path(config_loader.get('training.training_path'))
classifier.set_testing_set_path(config_loader.get('training.testing_path'))

classifier.set_layers([32, 64], [1000])
classifier.set_initial_deviation(0.03, 0.01)
