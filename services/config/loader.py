import yaml


class Loader:
    def __init__(self, path):
        self.configs = {}
        with open(path) as file:
            read_data = file.read()
            self.configs = yaml.load(read_data)

    def get(self, key, default=None):
        parsed_key = key.split('.')
        root = self.configs
        for part in parsed_key:
            try:
                root = root[part]
            except KeyError:
                return default

        return root
