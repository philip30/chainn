import numpy as np

import chainer.functions as F
import chainer.functions as L

from chainer import ChainList, Variable
from chainn import Vocabulary

class RNN(ChainList):
    def __init__(self, src_voc, trg_voc, input, output, hidden, depth, embed):
        super(RNN, self).__init__(
            *self._generate_layer(input, output, hidden, depth, embed)
        )
        self._input   = input
        self._output  = output
        self._hidden  = hidden
        self._depth   = depth
        self._embed   = embed
        self._h       = None
        self._src_voc = src_voc
        self._trg_voc = trg_voc

    def reset_state(self, xp, batch=1):
        hidden  = self._hidden
        self._h = Variable(xp.zeros((batch, hidden), dtype=np.float32))

    def __call__(self, word):
        if self._h is None:
            raise Exception("Need to call reset_state() before using the model!")
        embed  = self[0]
        e_to_h = self[1]
        h_to_h = self[2]
        h_to_y = self[-1]
        x = F.tanh(embed(word))
        h = F.tanh(e_to_h(x) + h_to_h(self._h))
        for i in range(3,len(self)-1):
            h = F.tanh(self[i](h))
        self._h = h
        y = F.tanh(h_to_y(h))
        return y

    def save(self, fp):
        fp.write(self._input)
        fp.write(self._output)
        fp.write(self._hidden)
        fp.write(self._depth)
        fp.write(self._embed)
        self._src_voc.save(fp)
        self._trg_voc.save(fp)
        fp.write_param_list(self)
  
    @staticmethod
    def load(fp, Model):
        input  = int(fp.read())
        output = int(fp.read())
        hidden = int(fp.read())
        depth  = int(fp.read())
        embed  = int(fp.read())
        src    = Vocabulary.load(fp)
        trg    = Vocabulary.load(fp)
        ret    = Model(src, trg, input, output, hidden, depth, embed)
        fp.read_param_list(ret)
        return ret

    # PROTECTED
    def _generate_layer(input, output, hidden, depth, embed):
        assert(depth >= 1)
        ret = []
        ret.append(L.EmbedID(input, embed))
        ret.append(L.Linear(embed, hidden))
        for _ in range(depth):
            ret.append(L.Linear(hidden, hidden))
        ret.append(L.Linear(hidden, output))
        return ret

