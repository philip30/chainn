import numpy as np

import chainer.functions as F
import chainer.functions as L

from chainer import ChainList, Variable
from chainn import Vocabulary

class RNN(ChainList):
    name = "RNN"

    def __init__(self, src_voc, trg_voc, input, output, hidden, depth, embed, activation=F.tanh):
        super(RNN, self).__init__(
            *self._generate_layer(input, output, hidden, depth, embed)
        )
        self._name    = RNN.name
        self._input   = input
        self._output  = output
        self._hidden  = hidden
        self._depth   = depth
        self._embed   = embed
        self._h       = None
        self._src_voc = src_voc
        self._trg_voc = trg_voc
        self._activation = activation

    def reset_state(self, xp, batch=1):
        hidden  = self._hidden
        self._h = Variable(xp.zeros((batch, hidden), dtype=np.float32))

    def __call__(self, word, update=True):
        if self._h is None:
            raise Exception("Need to call reset_state() before using the model!")
        embed  = self[0]
        e_to_h = self[1]
        h_to_h = self[2]
        h_to_y = self[-1]
        f = self._activation
        x = embed(word)
        h = e_to_h(x) + h_to_h(self._h)
        for i in range(3,len(self)-1):
            h = self[i](h)
        if update:
            self._h = h
        y = f(h_to_y(h))
        return y

    def save(self, fp):
        fp.write(self._name)
        fp.write("Inp:\t"+str(self._input))
        fp.write("Out:\t"+str(self._output))
        fp.write("Hid:\t"+str(self._hidden))
        fp.write("Dep:\t"+str(self._depth))
        fp.write("Emb:\t"+str(self._embed))
        fp.write_activation(self._activation)
        self._src_voc.save(fp)
        self._trg_voc.save(fp)
        fp.write_param_list(self)
  
    @staticmethod
    def load(fp, Model):
        input  = int(fp.read().split("\t")[1])
        output = int(fp.read().split("\t")[1])
        hidden = int(fp.read().split("\t")[1])
        depth  = int(fp.read().split("\t")[1])
        embed  = int(fp.read().split("\t")[1])
        act    = fp.read_activation()
        src    = Vocabulary.load(fp)
        trg    = Vocabulary.load(fp)
        ret    = Model(src, trg, input, output, hidden, depth, embed, act)
        fp.read_param_list(ret)
        return ret

    # PROTECTED
    def _generate_layer(self, input, output, hidden, depth, embed):
        assert(depth >= 1)
        ret = []
        ret.append(L.EmbedID(input, embed))
        for i in range(depth+1):
            start = embed if i == 0 else hidden
            ret.append(L.Linear(start, hidden))
        ret.append(L.Linear(hidden, output))
        return ret

