import abc


class FileLoader:
    @abc.abstractmethod
    def load(self, string):
        pass
