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

classifier.initialize()
classifier.train()

classes = {
    0: 'tongue',
    1: 'no tongue',
}

to_classify = [
    'https://www.apollonia-dcc.com/wp-content/uploads/2015/10/1.jpg',
    'https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/The_Earth_seen_from_Apollo_17.jpg/1280px-The_Earth_seen_from_Apollo_17.jpg',  # noqa
    'https://www.rugbywarfare.com/store/wp-content/uploads/2017/04/random-image-005.jpg',
    'https://dw9to29mmj727.cloudfront.net/promo/2016/5257-SeriesHeaders_SMv3_2000x800.jpg',
    'https://cbsnews2.cbsistatic.com/hub/i/r/2011/02/23/e519b0f1-a642-11e2-a3f0-029118418759/resize/620x465/6e3e7b4b05105fcc86ddb2d0a4c7c626/Black-hairy-tongue.jpg',  # noqa
]

predictions = classifier.classify(to_classify)

for key, prediction in enumerate(predictions):
    print(f'\n{to_classify[key]} - {classes[prediction[0]]}, accuracy: {prediction[1]}%')
