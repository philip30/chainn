import numpy as np

import chainer.functions as F
import chainer.links as L

from chainer import Variable

from chainn import Vocabulary
from chainn.link import LSTM

from . import RNN

class LSTMRNN(RNN):
    name="lstm"

    def reset_state(self, *args, **kwargs):
        for item in self:
            if type(item) == LSTM:
                item.reset_state()

    def __call__(self, word, update=True, is_train=False):
        f = self._activation
        embed  = self[0]
        h_to_y = self[-1]
        x = embed(word)
        for i in range(1,len(self)-1):
            h = F.dropout(self[i](x if i == 1 else h, update), train=is_train)
        y = f(h_to_y(h))
        return y

    def _construct_model(self, input, output, hidden, depth, embed):
        assert(depth >= 1)
        ret = []
        ret.append(L.EmbedID(input, embed))
        for i in range(depth+1):
            start = embed if i == 0 else hidden
            ret.append(LSTM(start, hidden))
        ret.append(L.Linear(hidden, output))
        return ret

 
