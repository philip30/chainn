import numpy as np

import chainer.functions as F
import chainer.links as L

from chainer import Variable

from chainn import Vocabulary
from chainn.model import RNN

class LSTMRNN(RNN):
    def __init__(self, *args, **kwargs):
        super(LSTMRNN, self).__init__(*args, **kwargs)

    def reset_state(self, *args, **kwargs):
        for i in range(1, len(self)-1):
            self[i].reset_state()

    def __call__(self, word):
        embed  = self[0]
        h_to_y = self[-1]
        x = F.tanh(embed(word))
        for i in range(1,len(self)-1):
            h = F.tanh(self[i](x if i == 1 else h))
        y = F.tanh(h_to_y(h))
        return y

    def _generate_layer(self, input, output, hidden, depth, embed):
        assert(depth >= 1)
        ret = []
        ret.append(L.EmbedID(input, embed))
        for i in range(depth):
            start = embed if i == 0 else hidden
            ret.append(L.LSTM(start, hidden))
        ret.append(L.Linear(hidden, output))
        return ret

 
