import numpy as np

import chainer.functions as F
import chainer.links as L

from chainer import Variable

from chainn import Vocabulary
from chainn.model import RNN
from chainn.link import LSTM

class LSTMRNN(RNN):
    def __init__(self, *args, **kwargs):
        super(LSTMRNN, self).__init__(*args, **kwargs)

    def reset_state(self, *args, **kwargs):
        for item in self:
            if type(item) == LSTM:
                item.reset_state()

    def __call__(self, word, update=True):
        f = self._activation
        embed  = self[0]
        h_to_y = self[-1]
        x = embed(word)
        for i in range(1,len(self)-1):
            h = self[i](x if i == 1 else h, update)
        y = f(h_to_y(h))
        return y

    def _generate_layer(self, input, output, hidden, depth, embed):
        assert(depth >= 1)
        ret = []
        ret.append(L.EmbedID(input, embed))
        for i in range(depth):
            start = embed if i == 0 else hidden
            ret.append(LSTM(start, hidden))
        ret.append(L.Linear(hidden, output))
        return ret

 
