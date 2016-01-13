import chainn
import chainer
import chainer.functions as F
import numpy as np

from chainer import optimizers
from chainer.links.connection.linear import Linear
from chainer.links.connection.embed_id import EmbedID
from chainer.links.connection.lstm import LSTM

from chainn.link import LinearInterpolation, StackLSTM

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

    def write_vector(self, x):
        self.write(vtos(x))

    def write_matrix(self, x):
        for row in x:
            self.write_vector(row)

    def read(self):
        return next(self.__fp).strip()

    def read_vector(self, x, tp):
        data = stov(self.read(), tp)
        for i in range(len(data)):
            x[i] = data[i]

    def read_matrix(self, x, tp):
        for row in x:
            self.read_vector(row, tp)

    # Chainer Link Write
    def write_embed(self, f):
        self.write_matrix(f.W.data)

    def write_linear(self, f):
        self.write_matrix(f.W.data)
        self.write_vector(f.b.data)

    def write_lstm(self, f):
        self.write_matrix(f.upward.W.data)
        self.write_vector(f.upward.b.data)
        self.write_matrix(f.lateral.W.data)
    
    def write_linter(self, f):
        self.write(f.W.data[0])

    # Chainer Link Read
    def read_embed(self, f):
        self.read_matrix(f.W.data, float)

    def read_linear(self, f):
        self.read_matrix(f.W.data, float)
        self.read_vector(f.b.data, float)

    def read_lstm(self, f):
        self.read_matrix(f.upward.W.data, float)
        self.read_vector(f.upward.b.data, float)
        self.read_matrix(f.lateral.W.data, float)

    def read_linter(self, f):
        f.W.data[...] = float(self.read()) 

    def get_file_pointer(self):
        return self.__fp

    def read_param_list(self, param):
        for i, item in enumerate(param):
            if type(item) == Linear:
                self.read_linear(param[i])
            elif type(item) == EmbedID:
                self.read_embed(param[i])
            elif type(item) == LSTM or type(item) == chainn.link.LSTM:
                self.read_lstm(param[i])
            elif type(item) == LinearInterpolation:
                self.read_linter(param[i])
            elif type(item) == StackLSTM:
                self.read_param_list(param[i])
            else:
                raise NotImplementedError(type(item))

    def write_param_list(self, param):
        for item in param:
            if type(item) == Linear:
                self.write_linear(item)
            elif type(item) == EmbedID:
                self.write_embed(item)
            elif type(item) == LSTM or type(item) == chainn.link.LSTM:
                self.write_lstm(item)
            elif type(item) == LinearInterpolation:
                self.write_linter(item)
            elif type(item) == StackLSTM:
                self.write_param_list(item)
            else:
                raise NotImplementedError(type(item))

    def write_activation(self, f):
        if f == F.tanh:
            self.write("tanh")
        elif f == F.relu:
            self.write("relu")
        else:
            raise NotImplementedError(type(f))

    def read_activation(self):
        line = self.read()
        if line == "tanh": return F.tanh
        elif line == "relu": return F.relu
        else: raise NotImplementedError(type(f))

    def write_optimizer_state(self, opt):
        if type(opt) == optimizers.SGD:
            self.write("sgd\t%.30f" % opt.lr)
        elif type(opt) == optimizers.AdaDelta:
            self.write("adadelta")
        elif type(opt) == optimizers.AdaGrad:
            self.write("adagrad\t%.30f" %opt.lr)
        else:
            raise NotImplementedError(type(opt))

    def read_optimizer_state(self):
        line = self.read().split("\t")
        opt = None
        if line[0] == "sgd":
            opt = optimizers.SGD(lr=float(line[1]))
        elif line[0] == "adadelta":
            opt = optimizers.AdaDelta()
        elif line[0] == "adagrad":
            opt = optimizers.AdaGrad(lr=float(line[1]))
        else:
            raise NotImplementedError(line[0])
        return opt

