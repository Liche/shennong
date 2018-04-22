import tensorflow as tf

from contextlib import contextmanager
from services.file.file_loader import FileLoader


class LocalLoader(FileLoader):
    @contextmanager
    def load(self, string):
        yield tf.read_file(string)

