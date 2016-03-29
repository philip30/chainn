import chainn
import chainer
import chainer.functions as F
from chainn.util import functions as UF
import numpy as np

from chainer import optimizers, ChainList, Variable
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
        if x is None: return
        if type(x) == Variable:
            x = x.data
        self.write(vtos(x))

    def write_matrix(self, x):
        if x is None: return
        x = x.data
        for row in x:
            self.write_vector(row)

    def write_2leveldict(self, dct):
        total = sum(len(value) for key, value in dct.items())
        self.write(total)
        for key, value in sorted(dct.items()):
            for key2, value in sorted(value.items()):
                self.write(str(key) + "\t" + str(key2) + "\t" + str(value))

    def read(self):
        return next(self.__fp).strip()

    def read_vector(self, x, tp):
        if x is None: return
        data = stov(self.read(), tp)
        x = x.data
        for i in range(len(data)):
            x[i] = data[i]

    def read_matrix(self, x, tp):
        if x is None: return
        x = x.data
        for row in x:
            self.read_vector(row, tp)
    
    def read_2leveldict(self, dct):
        number = int(self.read())
        for i in range(number):
            line = self.read().split("\t")
            dct[int(line[0])][int(line[1])] = float(line[2])

    # Chainer Link Write
    def write_embed(self, f):
        self.write_matrix(f.W)

    def write_linear(self, f):
        self.write_matrix(f.W)
        self.write_vector(f.b)

    def write_lstm(self, f):
        self.write_matrix(f.upward.W)
        self.write_vector(f.upward.b)
        self.write_matrix(f.lateral.W)
    
    def write_linter(self, f):
        self.write(f.W.data[0])

    # Chainer Link Read
    def read_embed(self, f):
        UF.trace("Reading Embed", debug_level=1)
        self.read_matrix(f.W, float)

    def read_linear(self, f):
        UF.trace("Reading Linear", debug_level=1)
        self.read_matrix(f.W, float)
        self.read_vector(f.b, float)

    def read_lstm(self, f):
        UF.trace("Reading LSTM", debug_level=1)
        self.read_matrix(f.upward.W, float)
        self.read_vector(f.upward.b, float)
        self.read_matrix(f.lateral.W, float)

    def read_linter(self, f):
        UF.trace("Reading Linear Interpolation", debug_level=1)
        f.W.data[...] = float(self.read()) 

    def get_file_pointer(self):
        return self.__fp

    def read_param_list(self, param):
        UF.trace("Reading Param List", debug_level=1)
        for i, item in enumerate(param):
            if type(item) == Linear:
                self.read_linear(param[i])
            elif type(item) == EmbedID:
                self.read_embed(param[i])
            elif type(item) == LSTM:
                self.read_lstm(param[i])
            elif type(item) == LinearInterpolation:
                self.read_linter(param[i])
            elif issubclass(item.__class__, ChainList):
                self.read_param_list(param[i])
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
            elif type(item) == LinearInterpolation:
                self.write_linter(item)
            elif issubclass(item.__class__, ChainList):
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
        UF.trace("Reading Activation", debug_level=1)
        line = self.read()
        if line == "tanh": return F.tanh
        elif line == "relu": return F.relu
        else: raise NotImplementedError(type(f))

    def write_optimizer_state(self, opt):
        if type(opt) == optimizers.SGD:
            self.write("sgd\t%f" % opt.lr)
        elif type(opt) == optimizers.AdaDelta:
            self.write("adadelta\t%f\t%f" % (opt.rho, opt.eps))
        elif type(opt) == optimizers.AdaGrad:
            self.write("adagrad\t%f" %opt.lr)
        elif type(opt) == optimizers.Adam:
            self.write("adam\t%f\t%f\t%f\t%f" % (opt.alpha, opt.beta1, opt.beta2, opt.eps))
        else:
            raise NotImplementedError(type(opt))

    def read_optimizer_state(self):
        UF.trace("Reading Optimizer State", debug_level=1)
        line = self.read().split("\t")
        opt = None
        if line[0] == "sgd":
            opt = optimizers.SGD(lr=float(line[1]))
        elif line[0] == "adadelta":
            opt = optimizers.AdaDelta(rho=float(line[1]),eps=float(line[2]))
        elif line[0] == "adagrad":
            opt = optimizers.AdaGrad(lr=float(line[1]))
        elif line[0] == "adam":
            opt = optimizers.Adam(alpha=float(line[1]), beta1=float(line[2]), beta2=float(line[3]), eps=float(line[4]))
        else:
            raise NotImplementedError(line[0])
        return opt

