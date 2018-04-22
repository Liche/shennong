import os
import tensorflow as tf
import tempfile
import requests

from contextlib import contextmanager
from services.file.file_loader import FileLoader


class UrlLoader(FileLoader):
    @contextmanager
    def load(self, string):
        with self.as_file(string) as file:
            yield tf.read_file(file.name)
            file.close()
            os.remove(file.name)

    @contextmanager
    def as_file(self, url):
        with tempfile.NamedTemporaryFile(delete=False) as file:
            file.write(requests.get(url).content)
            file.flush()
            yield file
