import chainer
import numpy as np

from chainer.links.connection.linear import Linear
from chainer.links.connection.embed_id import EmbedID
from chainer.links.connection.lstm import LSTM

from collections import defaultdict

def vtos(v, fmt='%.8e'):
    return ' '.join(fmt % x for x in v)

def stov(s, tp=float):
    return [tp(x) for x in s.split()]

# Copied and modified from https://github.com/odashi/chainer_examples/blob/master/util/model_file.py
class ModelFile:
    def __init__(self, fp):
        self.__fp = fp

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__fp.close()
        return False

    def write(self, x):
        print(x, file=self.__fp)

    def __write_vector(self, x):
        self.write(vtos(x))

    def __write_matrix(self, x):
        for row in x:
            self.__write_vector(row)

    def read(self):
        return next(self.__fp).strip()

    def __read_vector(self, x, tp):
        data = stov(self.read(), tp)
        for i in range(len(data)):
            x[i] = data[i]

    def __read_matrix(self, x, tp):
        for row in x:
            self.__read_vector(row, tp)

    # Chainer Link Write
    def write_embed(self, f):
        self.__write_matrix(f.W.data)

    def write_linear(self, f):
        self.__write_matrix(f.W.data)
        self.__write_vector(f.b.data)

    def write_lstm(self, f):
        self.__write_matrix(f.upward.W.data)
        self.__write_vector(f.upward.b.data)
        self.__write_matrix(f.lateral.W.data)
    
    # Chainer Link Read
    def read_embed(self, f):
        self.__read_matrix(f.W.data, float)

    def read_linear(self, f):
        self.__read_matrix(f.W.data, float)
        self.__read_vector(f.b.data, float)

    def read_lstm(self, f):
        self.__read_matrix(f.upward.W.data, float)
        self.__read_vector(f.upward.b.data, float)
        self.__read_matrix(f.lateral.W.data, float)

    def get_file_pointer(self):
        return self.__fp

    def read_param_list(self, param):
        for i, item in enumerate(param):
            if type(item) == Linear:
                self.read_linear(param[i])
            elif type(item) == EmbedID:
                self.read_embed(param[i])
            elif type(item) == LSTM:
                self.read_lstm(param[i])
            else:
                raise NotImplementedError(type(item))

    def write_param_list(self, param):
        for item in param:
            if type(item) == Linear:
                self.write_linear(item)
            elif type(item) == EmbedID:
                self.write_embed(item)
            elif type(item) == LSTM:
                self.write_lstm(item)
            else:
                raise NotImplementedError(type(item))
