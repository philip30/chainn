import chainer.functions as F
import chainer.links as L

from chainn import Vocabulary
from chainer import ChainList, Variable

# Multi Layer Perceptron
class MLP(ChainList):
    def __init__(self, feat, trg, input, output, hidden, depth):
        super(MLP, self).__init__(*generate_layer(input, output, hidden, depth))
        self._input  = input
        self._output = output
        self._hidden = hidden
        self._depth  = depth
        self._feat   = feat
        self._trg    = trg

    def __call__(self, x):
        h = F.tanh(self[0](x))
        for i in range(1,len(self)):
            h = F.tanh(self[i](h))
        return h

    def save(self, fp):
        fp.write(self._input)
        fp.write(self._output)
        fp.write(self._hidden)
        fp.write(self._depth)
        self._feat.save(fp)
        self._trg.save(fp)
        fp.write_param_list(self)
  
    @staticmethod
    def load(fp):
        input  = int(fp.read())
        output = int(fp.read())
        hidden = int(fp.read())
        depth  = int(fp.read())
        feat   = Vocabulary.load(fp)
        trg    = Vocabulary.load(fp)
        ret    = MLP(feat, trg, input, output, hidden, depth)
        fp.read_param_list(ret)
        return ret

def generate_layer(input, output, hidden, depth):
    ret = []
    start = input
    end = hidden
    if depth <= 0:
        end = start
    else:
        ret.append(L.Linear(start, hidden))
        for _ in range(depth-1):
            ret.append(L.Linear(hidden, hidden))
    ret.append(L.Linear(end, output))
    return ret

